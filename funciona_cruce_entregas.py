# cruce_entregas_capita.py
"""
Cruce y agregación de entregas por medicamento (eventos por mes) y opcionalmente Capita.
- Preserva el orden y columnas del target.csv si lo proporcionas.
- Matching exhaustivo: token overlap, Jaccard, SequenceMatcher/rapidfuzz, keys (codigo/cum).
- Clasifica entregas: TOTAL, PARCIAL, PENDIENTE (usa CANTIDAD_SOLICITADA cuando exista).
- Maneja múltiples archivos evento (ej. AbrilEvento.csv, MayoEvento.csv, ...) y 0..N Capita files.
- Salidas:
    - entregas_por_meses_resumen_tipos.csv   (fila por target, meses como columnas, totales)
    - matched_audit_{timestamp}.csv          (fila por match usado para auditoría)
"""

import re
import sys
import argparse
import logging
from pathlib import Path
from decimal import Decimal, getcontext
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import pandas as pd # type: ignore
import numpy as np # type: ignore
import datetime
import psutil # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm # type: ignore

getcontext().prec = 28

# Configuración de los umbrales de puntuación para el emparejamiento
STRICT = {
    'MIN_ACCEPT_SCORE': 95, 
    'MIN_SCORE_GAP': 5,
    'MIN_JACCARD': 0.8
}
RELAXED = {
    'MIN_ACCEPT_SCORE': 85, 
    'MIN_SCORE_GAP': 2,
    'MIN_JACCARD': 0.7
}

# ---------------- Intentar usar rapidfuzz si está disponible (más rápido) ----------------
try:
    from rapidfuzz import fuzz # type: ignore
    _HAS_RAPIDFUZZ = True
    logging.info("Usando rapidfuzz para un cálculo de similitud más rápido.")
except Exception:
    fuzz = None
    _HAS_RAPIDFUZZ = False
    logging.warning("rapidfuzz no está instalado, se usará difflib.SequenceMatcher. La ejecución puede ser más lenta.")

# ---------------- Config por defecto (ajustables) ----------------
ENCODINGS = ["utf-8", "latin-1", "cp1252"]
SEPS = [';', ',', '\t', '|']

# Por defecto: sin límite (None) — se puede controlar por CLI o editar aquí.
MAX_CANDIDATES_EVAL = None
TOKEN_POOL_TOPK = None
TOP_N_FALLBACK = 3   # si no hay aceptados, devolver top-N para auditoría

# Se ajustan los pesos para el nuevo puntaje híbrido
WEIGHTS = {
    'desc_exact': 220,
    'desc_substring': 150,
    'jaccard': 110,
    'seq_ratio': 70,
    'token_sort_ratio': 150, # NUEVO: Peso para token_sort_ratio
    'token_common': 12,
    'same_codigo_neg': 70,
    'same_cum': 50,
    'presentation_match': 25,
    'cantidad_exists': 5
}

PRESENTATION_TERMS = {
    'tableta','tabletas','tablet','comprimido','comprimidos','capsula','capsulas','ampolla','ampollas',
    'caja','frasco','ml','mg','g','mcg','ug','tubo','sobre','solucion','suspension','inhalador','crema','gel','jarabe',
    'supositorio','ovulo','ampolleta','spray','unidad','unidades'
}

# tolerancia relativa para considerar entregado == solicitado (por defecto 1%)
TOLERANCE_RELATIVE = Decimal('0.01')

# ---------------- Utilidades ----------------
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

def read_csv_robust(path: Path) -> pd.DataFrame:
    last_exc = None
    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python', header=0)
                if df.shape[1] >= 2:
                    logging.debug(f"Leido {path} con encoding={enc} sep='{sep}'")
                    return df
            except Exception as e:
                last_exc = e
    try:
        df = pd.read_csv(path, engine='python', on_bad_lines='skip')
        return df
    except Exception:
        raise last_exc if last_exc is not None else Exception(f"No se pudo leer {path}")

def normalize_key(s):
    if pd.isna(s): return ""
    s = str(s).strip().upper()
    s = s.replace(',', '.')
    s = re.sub(r'\.0+$', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def normalize_desc(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\s\W]+", " ", s)
    return s.strip()

_word_re = re.compile(r'\w+')
def tokens_from_text(s, min_len=2):
    if not s: return []
    toks = _word_re.findall(s.lower())
    return [t for t in toks if len(t) >= min_len]

_THOUSANDS_CHARS_RE = re.compile(r"[\'\s\u00A0\u202F\u2007\u2009]")
_CURRENCY_CHARS_RE = re.compile(r'[^\d\.,\-\(\)eE]')

def parse_decimal(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return None
    s = str(val).strip()
    if s == "": return None
    negative = False
    if s.startswith('(') and s.endswith(')'):
        negative = True; s = s[1:-1].strip()
    s = _THOUSANDS_CHARS_RE.sub('', s)
    s = _CURRENCY_CHARS_RE.sub('', s)
    if s.count('-') > 0:
        s = s.replace('-', ''); negative = True
    if s == "": return None
    try:
        if re.search(r'[eE]', s):
            d = Decimal(s)
        else:
            last_dot = s.rfind('.'); last_comma = s.rfind(',')
            if last_dot != -1 and last_comma != -1:
                if last_comma > last_dot:
                    s2 = s.replace('.', '').replace(',', '.')
                else:
                    s2 = s.replace(',', '')
            elif last_comma != -1:
                parts = s.split(',')
                if len(parts[-1]) == 3 and all(len(p) <= 3 for p in parts[:-1]):
                    s2 = ''.join(parts)
                else:
                    s2 = s.replace(',', '.')
            else:
                s2 = s
            if s2.count('.') > 1:
                parts = s2.split('.'); decimal_part = parts[-1]; int_part = ''.join(parts[:-1])
                s2 = int_part + '.' + decimal_part
            d = Decimal(s2)
        return -d if negative else d
    except Exception:
        try:
            return Decimal(str(float(s.replace(',','.'))))
        except Exception:
            return None

def jaccard_similarity(a_tokens, b_tokens):
    a = set(a_tokens); b = set(b_tokens)
    if not a and not b: return 0.0
    inter = a & b; union = a | b
    return len(inter)/len(union) if union else 0.0

def seq_ratio(a, b):
    """
    Ratio 0..1. Prefer rapidfuzz if está instalado, fallback a difflib.
    """
    if not a and not b: return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            return float(fuzz.ratio(a, b) / 100.0)
        except Exception:
            return SequenceMatcher(None, a, b).ratio()
    else:
        return SequenceMatcher(None, a, b).ratio()

# NUEVA FUNCIÓN: calcula la similitud de token ordenado
def token_sort_ratio(a, b):
    if not a and not b: return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            return float(fuzz.token_sort_ratio(a, b) / 100.0)
        except Exception:
            return 0.0 # si rapidfuzz falla, devolvemos 0
    else:
        return 0.0

# ---------------- Preprocesado y detección de columnas ----------------
def detect_columns(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    def find(patterns):
        for p in patterns:
            for c in cols:
                if p.lower() in c.lower():
                    return c
        return None
    mapping = {
        'codigo_neg': find(['codigo_negociado','codigo negociado','codigo_neg','cod_neg', 'codigo_ap', 'ap']),
        'codigo_cum': find(['codigo_cum','cum','codigo cum']),
        'nombre': find(['nombre_producto','nombre producto','descripcion_dci','descripcion dci','descripcion','producto','nombre']),
        'pa': find(['principio_activo','principio activo','dci','activo','principio']),
        'cantidad': find(['cantidad_entregada','cantidad','cantidad entregada','entregado','cantidad_entrega']),
        'cantidad_solicitada': find(['cantidad_solicitada','cantidad solicitada','cantidad_pedida','cantidad pedido','cantidad_ped']),
        'tipo': find(['tipo de entrega','tipo_entrega','tipo entrega','tipo','estado_entrega','estado'])
    }
    return mapping

def preprocess_df(df: pd.DataFrame, mapping=None) -> (pd.DataFrame, dict): # type: ignore
    df = df.copy()
    if mapping is None:
        mapping = detect_columns(df)
    # normalize columns
    df['_codigo_neg_norm'] = df[mapping['codigo_neg']].apply(normalize_key) if mapping['codigo_neg'] else ""
    df['_cum_norm'] = df[mapping['codigo_cum']].apply(normalize_key) if mapping['codigo_cum'] else ""
    df['_nombre_norm'] = df[mapping['nombre']].apply(normalize_desc) if mapping['nombre'] else ""
    df['_pa_norm'] = df[mapping['pa']].apply(normalize_desc) if mapping['pa'] else ""
    df['_tokens'] = df['_nombre_norm'].apply(tokens_from_text)
    df['_presentation_tokens'] = df['_tokens'].apply(lambda t: set([x for x in t if x in PRESENTATION_TERMS]))
    df['_token_set'] = df['_tokens'].apply(lambda t: frozenset(t) if isinstance(t, list) else frozenset())
    df['_cantidad_parsed'] = df[mapping['cantidad']].apply(parse_decimal) if mapping['cantidad'] else None
    df['_cantidad_solicitada_parsed'] = df[mapping['cantidad_solicitada']].apply(parse_decimal) if mapping['cantidad_solicitada'] else None
    df['_tipo_entrega_raw'] = df[mapping['tipo']].fillna("").astype(str) if mapping['tipo'] else ""
    return df, mapping

# ---------------- Índices para búsqueda (por archivo) ----------------
def build_token_index(df):
    token_to_indices = defaultdict(list)
    for idx, toks in df['_token_set'].items():
        for tk in toks:
            token_to_indices[tk].append(idx)
    idx_by_codigo_neg = {k: g.index.tolist() for k,g in df.groupby('_codigo_neg_norm', sort=False)}
    idx_by_cum = {k: g.index.tolist() for k,g in df.groupby('_cum_norm', sort=False)}
    idx_by_pa = {k: g.index.tolist() for k,g in df.groupby('_pa_norm', sort=False)}
    return token_to_indices, idx_by_codigo_neg, idx_by_cum, idx_by_pa

# ---------------- Scoring y selección (idéntico estilo, optimizado) ----------------
# Se agrega la nueva métrica en el score_candidate_target
def score_candidate_target(target_desc_norm, target_pa_norm, target_codigo_neg, target_cum, df, idx):
    pr = df.loc[idx]
    pr_desc = pr.get('_nombre_norm','') or ''
    pr_pa = pr.get('_pa_norm','') or ''
    pr_codigo_neg = pr.get('_codigo_neg_norm', '') or ''
    pr_cum = pr.get('_cum_norm', '') or ''
    
    score = 0.0
    reasons = []

    # PRIORIDAD 1: Coincidencia exacta de código negociado
    if target_codigo_neg and target_codigo_neg == pr_codigo_neg:
        score += 1000  # Puntuación muy alta para coincidencia exacta
        reasons.append('exact_codigo_neg')
    
    # PRIORIDAD 2: Coincidencia exacta de CUM
    elif target_cum and target_cum == pr_cum:
        score += 900  # Puntuación alta para coincidencia exacta de CUM
        reasons.append('exact_cum')
    
    # PRIORIDAD 3: Coincidencias parciales o textuales (solo si no hay códigos)
    else:
        t_tokens = set(tokens_from_text(target_desc_norm))
        p_tokens = set(pr.get('_tokens',[]) or [])
        
        # New hybrid scoring (solo si no tenemos códigos)
        score_hybrid = 0.0
        sim_jaccard = jaccard_similarity(t_tokens, p_tokens)
        sim_seq_ratio = seq_ratio(target_desc_norm, pr_desc)
        sim_token_sort = token_sort_ratio(target_desc_norm, pr_desc)
        
        # Nuevo puntaje híbrido ponderado: 40% Token Sort, 30% Fuzz, 30% Jaccard
        score_hybrid = (0.4 * sim_token_sort) + (0.3 * sim_seq_ratio) + (0.3 * sim_jaccard)
        score += score_hybrid * 10  # Escalar para que sea comparable
        
        reasons.append(f'hybrid:{score_hybrid:.3f}')
        reasons.append(f'jacc:{sim_jaccard:.3f}')
        reasons.append(f'seq:{sim_seq_ratio:.3f}')
        reasons.append(f'token_sort:{sim_token_sort:.3f}')

        # Sumar otros pesos que no sean métricas de similitud de texto
        if target_desc_norm and pr_desc and target_desc_norm == pr_desc:
            score += WEIGHTS['desc_exact']
            reasons.append('desc_exact')
        elif target_desc_norm and pr_desc and (target_desc_norm in pr_desc or pr_desc in target_desc_norm):
            score += WEIGHTS['desc_substring']
            reasons.append('desc_substring')

    # Verificación de principio activo (añade puntos adicionales)
    if target_pa_norm and target_pa_norm == pr_pa:
        score += WEIGHTS['same_cum']
        reasons.append('same_pa')

    # Puntos por tener cantidad
    if pr.get('_cantidad_parsed') is not None:
        score += WEIGHTS['cantidad_exists']
        reasons.append('has_cantidad')

    return score, {'score': score, 'reasons': reasons, 'jaccard': sim_jaccard if 'sim_jaccard' in locals() else 0, 
                  'seq_ratio': sim_seq_ratio if 'sim_seq_ratio' in locals() else 0, 
                  'token_sort': sim_token_sort if 'sim_token_sort' in locals() else 0, 
                  'cantidad': pr.get('_cantidad_parsed')}

def build_candidate_pool_for_target(df, token_to_indices, idx_by_codigo_neg, idx_by_cum, target_tokens, target_codigo_neg="", target_cum=""):
    """
    Construye la lista de índices candidatos en orden de prioridad.
    Respeta TOKEN_POOL_TOPK / MAX_CANDIDATES_EVAL cuando no son None.
    Si son None -> toma todos los candidatos posibles (primero por tokens, luego el resto).
    """
    pool = []
    if target_codigo_neg and target_codigo_neg in idx_by_codigo_neg:
        pool.extend(idx_by_codigo_neg[target_codigo_neg])
    if target_cum and target_cum in idx_by_cum:
        pool.extend(idx_by_cum[target_cum])
    if target_tokens:
        counter = Counter()
        for tk in target_tokens:
            for idx in token_to_indices.get(tk, []):
                counter[idx] += 1
        if counter:
            if TOKEN_POOL_TOPK is None:
                top_by_tokens = [idx for idx, _ in counter.most_common()]
            else:
                top_by_tokens = [idx for idx, _ in counter.most_common(TOKEN_POOL_TOPK)]
            for idx in top_by_tokens:
                if idx not in pool:
                    pool.append(idx)
    # completar hasta límite (si lo hay); si MAX_CANDIDATES_EVAL es None -> añadir todos
    for idx in df.index:
        if idx not in pool:
            pool.append(idx)
        if MAX_CANDIDATES_EVAL is not None and len(pool) >= MAX_CANDIDATES_EVAL:
            break
    return pool

def search_matches_for_target_in_df(df, token_to_indices, idx_by_codigo_neg, idx_by_cum,
                                    target_desc, target_pa,
                                    target_codigo_neg=None, target_cum=None,
                                    mode='strict', strict_cfg=None, relaxed_cfg=None, **kwargs):
    """
    Versión tolerante: acepta parámetros por varios nombres y kwargs extras.
    Soporta llamadas con:
      - target_codigo_neg / codigo_neg_norm / codigo_neg
      - target_cum / cum_norm / codigo_cum / cum
    """

    # --- Compatibilidad con diferentes nombres que puedan pasarse como kwargs ---
    if not target_codigo_neg:
        target_codigo_neg = kwargs.get('target_codigo_neg') or kwargs.get('codigo_neg_norm') or kwargs.get('codigo_neg')
    if not target_cum:
        target_cum = kwargs.get('target_cum') or kwargs.get('cum_norm') or kwargs.get('codigo_cum') or kwargs.get('cum')

    # defensas contra None para configs
    if strict_cfg is None:
        strict_cfg = {'MIN_ACCEPT_SCORE': 0.8, 'MIN_SCORE_GAP': 0.05}
    if relaxed_cfg is None:
        relaxed_cfg = {'MIN_ACCEPT_SCORE': 0.65, 'MIN_SCORE_GAP': 0.0}

    # normalizar descripciones/pa y tokens
    t_desc_norm = normalize_desc(target_desc)
    t_pa_norm = normalize_desc(target_pa)
    t_tokens = set(tokens_from_text(t_desc_norm))

    # construir pool de candidatos (la función puede usar idx_by_codigo_neg/idx_by_cum)
    # le pasamos siempre los códigos estandarizados en los nombres que usemos internamente
    pool = build_candidate_pool_for_target(
        df, token_to_indices, idx_by_codigo_neg, idx_by_cum, t_tokens,
        target_codigo_neg=target_codigo_neg, target_cum=target_cum
    )

    cfg = strict_cfg if mode == 'strict' else relaxed_cfg
    scored = []

    for idx in pool:
        # pasar los códigos a la función de scoring (asegúrate de actualizar score_candidate_target si no lo hace)
        try:
            s, meta = score_candidate_target(t_desc_norm, t_pa_norm, target_codigo_neg, target_cum, df, idx)
        except TypeError:
            # fallback: si score_candidate_target aún no acepta códigos, llamamos la firma antigua
            s, meta = score_candidate_target(t_desc_norm, t_pa_norm, df, idx)
        if meta is None:
            meta = {}
        scored.append((idx, float(s), meta))

    if not scored:
        return []

    # conteo de tokens en común con defensas
    def common_tokens_count(idx):
        try:
            return len(set(df.at[idx, '_tokens']) & t_tokens)
        except Exception:
            return 0

    scored.sort(key=lambda x: (
        -x[1],
        -float(x[2].get('seq_ratio', 0)),
        -common_tokens_count(x[0])
    ))

    top_score = scored[0][1]
    second_score = scored[1][1] if len(scored) > 1 else -1e9

    results = []
    for idx, s, meta in scored:
        accept = False
        # Regla principal según cfg activo
        if s >= cfg.get('MIN_ACCEPT_SCORE', 0) and (s - second_score) >= cfg.get('MIN_SCORE_GAP', 0):
            accept = True
        # fallback: si alcanza el umbral relajado lo aceptamos
        elif s >= relaxed_cfg.get('MIN_ACCEPT_SCORE', 0):
            accept = True

        # adicional: si hay coincidencia exacta por codigo normalizado, forzamos aceptación
        try:
            src_codigo_neg = str(df.at[idx, '_codigo_neg_norm']) if '_codigo_neg_norm' in df.columns else ""
            src_cum = str(df.at[idx, '_cum_norm']) if '_cum_norm' in df.columns else ""
            if target_codigo_neg and src_codigo_neg and target_codigo_neg == src_codigo_neg:
                accept = True
            if target_cum and src_cum and target_cum == src_cum:
                accept = True
        except Exception:
            pass

        if accept:
            results.append((idx, s, meta))

    # Si no hay resultados aceptados, devolver un fallback con los top N candidatos
    if not results:
        try:
            TOP_N_FALLBACK
        except NameError:
            TOP_N_FALLBACK = 3
        results = scored[:TOP_N_FALLBACK]

    return results

# ---------------- Clasificación de fila (TOTAL / PARCIAL / PENDIENTE) ----------------
def classify_and_accumulate_row(row, tol=TOLERANCE_RELATIVE):
    """
    Devuelve dic con decimales: delivered_total, delivered_partial, pending, label
    """
    d = row.get('_cantidad_parsed')
    r = row.get('_cantidad_solicitada_parsed')
    tipo = str(row.get('_tipo_entrega_raw','') or '').strip().lower()
    delivered = Decimal(d) if d is not None else None
    requested = Decimal(r) if r is not None else None
    delivered_total = Decimal('0'); delivered_partial = Decimal('0'); pending = Decimal('0'); label = 'unknown'

    # textual priority
    if 'total' in tipo or 'complet' in tipo or 'entrega completa' in tipo:
        # if requested present, try to align, else assume delivered provided
        if requested is not None:
            if delivered is None:
                delivered_total = requested
            else:
                if requested != 0 and (abs(delivered - requested) / (requested if requested!=0 else Decimal('1'))) <= tol:
                    delivered_total = delivered
                else:
                    delivered_total = delivered
        else:
            delivered_total = delivered if delivered is not None else Decimal('0')
        label = 'total'
    elif 'parcial' in tipo or 'parci' in tipo:
        if delivered is not None:
            delivered_partial = delivered
            if requested is not None and requested > delivered:
                pending = requested - delivered
        else:
            if requested is not None:
                pending = requested
        label = 'partial'
    elif 'pendient' in tipo or 'pend' in tipo:
        if requested is not None:
            pending = requested
        else:
            pending = Decimal('0')
        label = 'pending'
    else:
        # infer numeric
        if requested is not None and delivered is not None:
            if requested == 0:
                delivered_total = delivered
                label = 'total' if delivered != 0 else 'unknown'
            else:
                diff = requested - delivered
                if delivered == requested or (abs(diff) / (requested if requested!=0 else Decimal('1'))) <= tol:
                    delivered_total = delivered
                    label = 'total'
                elif delivered < requested:
                    delivered_partial = delivered
                    pending = requested - delivered
                    label = 'partial'
                else:
                    # delivered > requested
                    delivered_total = requested
                    delivered_partial = delivered - requested
                    label = 'total_plus_extra'
        else:
            if delivered is not None:
                delivered_partial = delivered
                label = 'partial' if delivered != 0 else 'unknown'
            elif requested is not None:
                pending = requested
                label = 'pending'
            else:
                label = 'unknown'
    return {'delivered_total': delivered_total, 'delivered_partial': delivered_partial, 'pending': pending, 'label': label}

# NUEVO: Worker para procesar un target de forma paralela
def process_one_target_worker(tidx, trow, target_cols, month_data, capita_data, mode, strict_cfg, relaxed_cfg):
    out_row = {col: trow[col] for col in target_cols}
    desc = str(trow.get('descripcion') or trow.get('DESCRIPCION') or trow.get('nombre_producto') or trow.get('NOMBRE_PRODUCTO') or "")
    pa = str(out_row.get('PRINCIPIO_ACTIVO') or out_row.get('principio_activo') or "")

    # Extraer código negociado y CUM del target (variantes de nombre)
    codigo_neg = str(trow.get('codigo_negociado') or trow.get('CODIGO_NEGOCIADO') or trow.get('codigo_neg') or trow.get('CODIGO_NEG') or "")
    cum = str(trow.get('codigo_cum') or trow.get('CODIGO_CUM') or trow.get('cum') or trow.get('CUM') or "")

    # Normalizar los códigos (usa tu función normalize_key)
    codigo_neg_norm = normalize_key(codigo_neg) if codigo_neg else ""
    cum_norm = normalize_key(cum) if cum else ""

    # Incluyo los códigos en out_row por si quieres exportarlos
    out_row['codigo_negociado'] = codigo_neg
    out_row['codigo_negociado_norm'] = codigo_neg_norm
    out_row['codigo_cum'] = cum
    out_row['codigo_cum_norm'] = cum_norm

    total_ent_total = Decimal('0'); total_ent_partial = Decimal('0'); total_pending = Decimal('0')
    matched_info = {}
    audit_rows = []

    # events per month
    for mn, mdata in sorted(month_data.items()):
        df = mdata['df']; token_idx = mdata['token_index']
        idx_by_codigo_neg = mdata.get('idx_by_codigo_neg'); idx_by_cum = mdata.get('idx_by_cum')
        # Pasa las configuraciones y códigos normalizados a la función de búsqueda
        matches = search_matches_for_target_in_df(
            df, token_idx, idx_by_codigo_neg, idx_by_cum,
            desc, pa,
            codigo_neg_norm=codigo_neg_norm, cum_norm=cum_norm,
            mode=mode, strict_cfg=strict_cfg, relaxed_cfg=relaxed_cfg
        )
        month_total_total = Decimal('0'); month_total_partial = Decimal('0'); month_pending = Decimal('0')

        for idx, s, meta in matches:
            row = df.loc[idx]
            classif = classify_and_accumulate_row(row)
            month_total_total += classif['delivered_total']
            month_total_partial += classif['delivered_partial']
            month_pending += classif['pending']

            audit_rows.append({
                'target_row': int(tidx),
                'target_desc': desc,
                'target_pa': pa,
                'target_codigo_neg': codigo_neg,
                'target_codigo_neg_norm': codigo_neg_norm,
                'target_cum': cum,
                'target_cum_norm': cum_norm,
                'source_type': 'evento',
                'source_file': mdata.get('file'),
                'source_label': mdata.get('label'),
                'matched_index': int(idx),
                'match_score': float(s),
                'match_meta': str(meta),
                'cantidad_parsed': str(row.get('_cantidad_parsed')),
                'cantidad_solicitada_parsed': str(row.get('_cantidad_solicitada_parsed')),
                'clasificacion_row': classif['label']
            })

        label = mdata.get('label')
        out_row[f'cantidad_total_{label}'] = str(month_total_total) if month_total_total != 0 else ''
        out_row[f'cantidad_parcial_{label}'] = str(month_total_partial) if month_total_partial != 0 else ''
        out_row[f'cantidad_pendiente_{label}'] = str(month_pending) if month_pending != 0 else ''
        total_ent_total += month_total_total
        total_ent_partial += month_total_partial
        total_pending += month_pending
        matched_info[label] = [int(match[0]) for match in matches]

    # procesa capita files
    for cap_label, cdata in capita_data.items():
        df = cdata['df']; token_idx = cdata['token_index']
        idx_by_codigo_neg = cdata.get('idx_by_codigo_neg'); idx_by_cum = cdata.get('idx_by_cum')
        matches = search_matches_for_target_in_df(
            df, token_idx, idx_by_codigo_neg, idx_by_cum,
            desc, pa,
            codigo_neg_norm=codigo_neg_norm, cum_norm=cum_norm,
            mode=mode, strict_cfg=strict_cfg, relaxed_cfg=relaxed_cfg
        )
        cap_total_total = Decimal('0'); cap_total_partial = Decimal('0'); cap_pending = Decimal('0')

        for idx, s, meta in matches:
            row = df.loc[idx]
            classif = classify_and_accumulate_row(row)
            cap_total_total += classif['delivered_total']
            cap_total_partial += classif['delivered_partial']
            cap_pending += classif['pending']

            audit_rows.append({
                'target_row': int(tidx),
                'target_desc': desc,
                'target_pa': pa,
                'target_codigo_neg': codigo_neg,
                'target_codigo_neg_norm': codigo_neg_norm,
                'target_cum': cum,
                'target_cum_norm': cum_norm,
                'source_type': 'capita',
                'source_file': cdata.get('file'),
                'source_label': cap_label,
                'matched_index': int(idx),
                'match_score': float(s),
                'match_meta': str(meta),
                'cantidad_parsed': str(row.get('_cantidad_parsed')),
                'cantidad_solicitada_parsed': str(row.get('_cantidad_solicitada_parsed')),
                'clasificacion_row': classif['label']
            })

        out_row[f'capita_total_{cap_label}'] = str(cap_total_total) if cap_total_total != 0 else ''
        out_row[f'capita_parcial_{cap_label}'] = str(cap_total_partial) if cap_total_partial != 0 else ''
        out_row[f'capita_pendiente_{cap_label}'] = str(cap_pending) if cap_pending != 0 else ''
        total_ent_total += cap_total_total
        total_ent_partial += cap_total_partial
        total_pending += cap_pending
        matched_info[f'capita_{cap_label}'] = [int(match[0]) for match in matches]

    # totales acumulados
    out_row['total_entregado_total'] = str(total_ent_total) if total_ent_total != 0 else ''
    out_row['total_entregado_parcial'] = str(total_ent_partial) if total_ent_partial != 0 else ''
    out_row['total_pendiente'] = str(total_pending) if total_pending != 0 else ''
    out_row['matched_rows_per_source'] = str(matched_info)
    out_row['index'] = tidx # Para reordenar después

    return out_row, audit_rows

# ---------------- Operación principal: agregar por archivos (eventos) y opcionalmente Capita ----------------
MONTHS_MAP = {
    'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,'julio':7,'agosto':8,
    'septiembre':9,'setiembre':9,'octubre':10,'noviembre':11,'diciembre':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}
def detect_month_from_filename(fname):
    s = fname.lower()
    for k,v in MONTHS_MAP.items():
        if k in s:
            return v, k
    m = re.search(r'[^0-9](0?[1-9]|1[0-2])[^0-9]', '_' + s + '_')
    if m:
        return int(m.group(1)), m.group(1)
    return None, None

def aggregate(event_files, target_df=None, capita_files=None, mode='strict', out_csv='entregas_por_meses_resumen_tipos.csv', audit_csv_prefix=None):
    """
    event_files: list of paths (one per month/event)
    target_df: pandas DataFrame (optional). If None, build targets unique from event+capita union.
    capita_files: optional list of Capita file paths to also process (aggregated under label 'capita_X' per file)
    """
    # preprocess event files
    month_data = {}
    for f in event_files:
        p = Path(f)
        if not p.exists():
            logging.warning(f"Archivo no encontrado: {f} -> saltando")
            continue
        df_raw = read_csv_robust(p)
        df, mapping = preprocess_df(df_raw)
        token_idx, idx_by_codigo_neg, idx_by_cum, idx_by_pa = build_token_index(df)
        month_num, month_key = detect_month_from_filename(p.name)
        if month_num is None:
            # usar secuencia incremental con etiqueta filename
            month_num = 100 + len(month_data) + 1
            month_key = p.stem
        month_data[month_num] = {
            'df': df,
            'token_index': token_idx,
            'idx_by_codigo_neg': idx_by_codigo_neg,
            'idx_by_cum': idx_by_cum,
            'label': month_key,
            'file': str(p)
        }
        logging.info(f"Preprocesado evento: {p.name} -> label={month_key}, rows={len(df)}")

    # preprocess capita files if provistas
    capita_data = {}
    if capita_files:
        for f in capita_files:
            p = Path(f)
            if not p.exists():
                logging.warning(f"Capita file not found: {f} -> skipping")
                continue
            df_raw = read_csv_robust(p)
            df, mapping = preprocess_df(df_raw)
            token_idx, idx_by_codigo_neg, idx_by_cum, idx_by_pa = build_token_index(df)
            label = p.stem
            capita_data[label] = {
                'df': df,
                'token_index': token_idx,
                'idx_by_codigo_neg': idx_by_codigo_neg,
                'idx_by_cum': idx_by_cum,
                'file': str(p)
            }
            logging.info(f"Preprocesado Capita: {p.name} rows={len(df)}")

    # if no targets, build from union of event+capita unique name+pa
    if target_df is None:
        uniq = set()
        for m in list(month_data.values()) + list(capita_data.values()):
            tmp = m['df'][['_nombre_norm','_pa_norm']].drop_duplicates()
            for _, r in tmp.iterrows():
                uniq.add((r['_nombre_norm'], r['_pa_norm']))
        target_list = [{'DESCRIPCION_DCI': u[0], 'PRINCIPIO_ACTIVO': u[1]} for u in uniq]
        target_df = pd.DataFrame(target_list)
        logging.info(f"Generados targets automáticos: {len(target_df)}")
    else:
        # normalizar column names access
        if not isinstance(target_df, pd.DataFrame):
            raise ValueError("target_df debe ser un pandas.DataFrame o None")
        # PRESERVAR EL ORDEN ORIGINAL del target
        target_df = target_df.copy()
        target_df['_original_index'] = range(len(target_df))

    # Preparación para el procesamiento paralelo
    targets_iter = list(target_df.reset_index(drop=True).iterrows())
    n_cores = psutil.cpu_count(logical=True)
    logging.info(f"Usando {n_cores} núcleos para el procesamiento paralelo de targets.")

    out_rows = []
    audit_rows = []

    # IMPORTANTE: Aquí pasamos las configuraciones a la función del worker
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {
            executor.submit(process_one_target_worker, tidx, trow, target_df.columns.tolist(), month_data, capita_data, mode, STRICT, RELAXED): tidx
            for tidx, trow in targets_iter
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando targets", unit="target"):
            try:
                out_row, new_audit_rows = future.result()
                out_rows.append(out_row)
                audit_rows.extend(new_audit_rows)
            except Exception as e:
                logging.error(f"Error al procesar un target: {e}")
    
    # Ordenar la lista de diccionarios por el índice original
    out_rows_sorted = sorted(out_rows, key=lambda r: r.get("index", 0))

    # Crear el DataFrame de pandas a partir de la lista ordenada
    df_out = pd.DataFrame(out_rows_sorted)

    # Verificar si el DataFrame no está vacío y si la columna 'index' existe, y luego eliminarla.
    if not df_out.empty and 'index' in df_out.columns:
        df_out = df_out.drop(columns=['index'])
    
    # Si el target tenía índice original, restaurar el orden exacto
    if '_original_index' in target_df.columns:
        df_out = df_out.sort_values('_original_index').drop(columns=['_original_index'])
    
    # guardar outputs
    df_out.to_csv(out_csv, index=False, sep=';', encoding='utf-8-sig')
    logging.info(f"Guardado resumen principal: {out_csv} | targets: {len(df_out)}")

    # guardar auditoría
    if audit_csv_prefix is None:
        audit_csv_prefix = f"matched_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_df = pd.DataFrame(audit_rows)
    audit_path = f"{audit_csv_prefix}.csv"
    audit_df.to_csv(audit_path, index=False, sep=';', encoding='utf-8-sig')
    logging.info(f"Guardado audit CSV: {audit_path} | filas auditadas: {len(audit_df)}")

    return df_out, audit_df

# ---------------- CLI y ejemplo ----------------
def main(argv=None):
    """
    argv: list or None. Si es None, se usan sys.argv (comportamiento normal).
    Retorna (df_out, audit_df) para poder usarlo desde notebooks.
    """
    parser = argparse.ArgumentParser(description="Cruce entregas por medicamento (eventos por mes) y Capita.")
    parser.add_argument("--targets", help="Ruta a target.csv (opcional). Si no se da, se generan targets desde archivos.", default=None)
    parser.add_argument("--event_files", nargs='+', help="Archivos evento (uno por mes).", required=False)
    parser.add_argument("--capita_files", nargs='*', help="Archivos Capita (opcionales).", default=[])
    parser.add_argument("--out", help="Archivo CSV de salida.", default="entregas_por_meses_resumen_tipos.csv")
    parser.add_argument("--audit_prefix", help="Prefijo para audit CSV.", default=None)
    parser.add_argument("--mode", choices=['strict','relaxed'], default='strict')
    parser.add_argument("--tolerance_pct", type=float, default=1.0, help="Tolerancia relativa en porcentaje (default 1%).")
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--max_candidates", type=int, default=None, help="Máximo candidatos a evaluar por target (usa -1 para sin límite)")
    parser.add_argument("--token_pool_topk", type=int, default=None, help="Top-K por token (usa -1 para todos)")

    # Si argv fue explícitamente proporcionado (p. ej. desde notebook), úsalo (parse_args obligará los errores).
    if argv is not None:
        parsed = parser.parse_args(argv)
    else:
        # En entornos tipo notebook/colab preferimos parse_known_args para no morir por SystemExit
        parsed, unknown = parser.parse_known_args()

    # procesar opciones de límite globales (si se pasaron)
    global MAX_CANDIDATES_EVAL, TOKEN_POOL_TOPK
    if parsed.max_candidates is not None:
        if parsed.max_candidates < 0:
            MAX_CANDIDATES_EVAL = None
        else:
            MAX_CANDIDATES_EVAL = int(parsed.max_candidates)
    if parsed.token_pool_topk is not None:
        if parsed.token_pool_topk < 0:
            TOKEN_POOL_TOPK = None
        else:
            TOKEN_POOL_TOPK = int(parsed.token_pool_topk)

    # Si no se proporcionaron event_files, intentamos autodetectar archivos comunes en el cwd
    if not parsed.event_files:
        from glob import glob
        candidates = []
        candidates += glob('*evento*.csv') + glob('*Evento*.csv') + glob('*EVENTO*.csv')
        candidates += glob('*abril*.csv') + glob('*mayo*.csv') + glob('*202*.csv')
        # eliminar duplicados manteniendo orden
        seen = set(); candidates_filtered = []
        for c in candidates:
            if c not in seen:
                seen.add(c); candidates_filtered.append(c)
        if candidates_filtered:
            # informar pero continuar
            setup_logging(getattr(logging, parsed.loglevel.upper(), logging.INFO))
            logging.warning(f"No se recibieron --event_files. Autodetectados {len(candidates_filtered)} archivos de evento: {candidates_filtered}")
            parsed.event_files = candidates_filtered
        else:
            # si no hay archivos, fallamos con mensaje claro (no traceback enorme)
            parser.print_help()
            logging.error("No se encontraron --event_files ni archivos evento en el directorio actual. Pasa --event_files <archivo1> [archivo2 ...] o coloca archivos '*evento*.csv' en el cwd.")
            # devolver tupla vacía para notebooks o terminar con código 2 desde terminal
            if argv is None:
                sys.exit(2)
            else:
                return None, None

    # ahora configurar logging y tolerancia
    setup_logging(getattr(logging, parsed.loglevel.upper(), logging.INFO))
    global TOLERANCE_RELATIVE
    TOLERANCE_RELATIVE = Decimal(str(parsed.tolerance_pct/100))

    # advertencia sobre rapidfuzz
    if _HAS_RAPIDFUZZ:
        logging.info("rapidfuzz detectado: se usará para comparaciones difusas (más rápido).")
    else:
        logging.warning("rapidfuzz NO detectado: se usará difflib.SequenceMatcher (más lento). Recomiendo 'pip install rapidfuzz'.")

    # leer targets si se pasó
    target_df = None
    if parsed.targets:
        tpath = Path(parsed.targets)
        if not tpath.exists():
            logging.error(f"Targets file not found: {parsed.targets}")
            if argv is None:
                sys.exit(1)
            else:
                raise FileNotFoundError(parsed.targets)
        target_df = read_csv_robust(tpath)
        logging.info(f"Leido target.csv: {len(target_df)} filas. Se preservará su orden/columnas.")

    # Ejecutar la agregación principal
    logging.info("Iniciando agregación. Archivos evento: %s | Capita: %s", parsed.event_files, parsed.capita_files)
    if MAX_CANDIDATES_EVAL is None:
        logging.info("MAX_CANDIDATES_EVAL = None -> evaluando TODOS los candidatos (puede ser muy costoso).")
    else:
        logging.info("MAX_CANDIDATES_EVAL = %s", MAX_CANDIDATES_EVAL)
    if TOKEN_POOL_TOPK is None:
        logging.info("TOKEN_POOL_TOPK = None -> usando todos los candidatos encontrados por token.")
    else:
        logging.info("TOKEN_POOL_TOPK = %s", TOKEN_POOL_TOPK)

    df_out, audit_df = aggregate(parsed.event_files, target_df=target_df, capita_files=parsed.capita_files, mode=parsed.mode, out_csv=parsed.out, audit_csv_prefix=parsed.audit_prefix)

    logging.info("Proceso finalizado.")
    print(df_out.head(10).to_string(index=False))
    print(f"Resumen guardado en: {parsed.out} | Auditoría: {len(audit_df)} filas")

    return df_out, audit_df

if __name__ == "__main__":
    main()