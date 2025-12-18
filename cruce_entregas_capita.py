# cruce_entregas_capita.py
"""
Cruce y agregación de entregas por medicamento.
REFACTORIZADO:
- En lugar de iterar por target, iteramos por fila de source para garantizar que cada entrega
  se asigne a un ÚNICO target (el mejor match), evitando duplicación de cantidades.
- Mantiene la lógica de limpieza de laboratorios y preservación de términos.
- Genera el reporte final pivotado (targets x meses).
"""

import re
import sys
import argparse
import logging
from pathlib import Path
from decimal import Decimal, getcontext
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import pandas as pd 
import numpy as np 
import datetime
import psutil 
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

getcontext().prec = 28

# ---------------- Configuración ----------------
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

try:
    from rapidfuzz import fuzz, process, utils # type: ignore
    _HAS_RAPIDFUZZ = True
    logging.info("Usando rapidfuzz para un cálculo de similitud más rápido.")
except Exception:
    fuzz = None
    _HAS_RAPIDFUZZ = False
    logging.warning("rapidfuzz no está instalado, se usará difflib.SequenceMatcher. La ejecución puede ser más lenta.")

ENCODINGS = ["utf-8", "latin-1", "cp1252"]
SEPS = [';', ',', '\t', '|']

WEIGHTS = {
    'desc_exact': 220,
    'desc_substring': 150,
    'jaccard': 110,
    'seq_ratio': 70,
    'token_sort_ratio': 150,
    'token_common': 12,
    'same_codigo_neg': 1000,
    'same_cum': 900,
    'presentation_match': 25,
    'cantidad_exists': 5
}

PRESENTATION_TERMS = {
    'tableta','tabletas','tablet','comprimido','comprimidos','capsula','capsulas','ampolla','ampollas',
    'caja','frasco','ml','mg','g','mcg','ug','tubo','sobre','solucion','suspension','inhalador','crema','gel','jarabe',
    'supositorio','ovulo','ampolleta','spray','unidad','unidades'
}

LABORATORIES = {
    'TECNOQUIMICAS', 'SIEGFRIED', 'FARMACAPSULAS', 'SANOFI AVENTIS', 'SANOFI', 'GRUNENTHAL',
    'LAPROFF', 'PROCAPS', 'NOVAMED', 'WINTHROP', 'ASTRA ZENECA', 'TECNOFARMA', 'COLMED',
    'ECAR', 'LASANTE', 'GENFAR', 'AG', 'PFIZER', 'GSK', 'GLAXOSMITHKLINE', 'NOVARTIS',
    'ROCHE', 'ABBOTT', 'BAYER', 'MERCK', 'BOEHRINGER', 'JANSSEN', 'MSD', 'LILLY',
    'BRISTOL', 'AMGEN', 'GINEF', 'MEMPHIS', 'COASPHARMA', 'AMERICAN GENERICS', 'HUMAX',
    'VITALIS', 'BLAU', 'BAXTER', 'B BRAUN', 'BIOSIDUS', 'MK', 'LA SANTE', 'LAFRANCOL',
    'BIOCHEM', 'ANGLOPHARMA', 'ESPECIALIDADES FARMACEUTICAS', 'JGB', 'QUIMIDROGAS',
    'PHARMACIDER', 'HOSPIMEDIKS', 'FRESENIUS', 'KABI', 'ITALCHEM', 'BEST', 'AMERICAN',
    'ANGLO', 'PHARMA', 'CORPAUL'
}

SORTED_LABS = sorted(list(LABORATORIES), key=len, reverse=True)
LABS_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, SORTED_LABS)) + r')\b', re.IGNORECASE)

TOLERANCE_RELATIVE = Decimal('0.01')
MAX_CANDIDATES_EVAL = None # Not strictly used in new arch but kept for compat
TOKEN_POOL_TOPK = 50       # Limit search space for performance

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

def clean_labs_and_extra_info(s):
    if pd.isna(s): return ""
    s_upper = str(s).upper()
    if '|' in s_upper:
        s_upper = s_upper.split('|')[0]
    s_clean = LABS_PATTERN.sub('', s_upper)
    s_clean = re.sub(r'\s+', ' ', s_clean).strip()
    return s_clean

def normalize_desc(s):
    if pd.isna(s): return ""
    s = clean_labs_and_extra_info(s)
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
    """
    Parses decimal numbers assuming Colombian/LatAm format:
    - Thousands separator: '.' (dot)
    - Decimal separator: ',' (comma)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)): return None
    s = str(val).strip()
    if s == "": return None

    negative = False
    if s.startswith('(') and s.endswith(')'):
        negative = True; s = s[1:-1].strip()
    if s.count('-') > 0:
        s = s.replace('-', ''); negative = True

    s = _THOUSANDS_CHARS_RE.sub('', s)
    s = _CURRENCY_CHARS_RE.sub('', s)
    if s == "": return None

    try:
        # Strategy: Remove thousands (dot), replace decimal (comma) with dot
        s2 = s.replace('.', '')
        s2 = s2.replace(',', '.')
        d = Decimal(s2)
        return -d if negative else d
    except Exception:
        # Fallback to float conversion if something weird happens
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
    if not a and not b: return 0.0
    if _HAS_RAPIDFUZZ:
        return float(fuzz.ratio(a, b) / 100.0)
    else:
        return SequenceMatcher(None, a, b).ratio()

def token_sort_ratio(a, b):
    if not a and not b: return 0.0
    if _HAS_RAPIDFUZZ:
        return float(fuzz.token_sort_ratio(a, b) / 100.0)
    else:
        return 0.0

# ---------------- Lógica de Columnas ----------------
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

    df['_codigo_neg_norm'] = df[mapping['codigo_neg']].apply(normalize_key) if mapping['codigo_neg'] else ""
    df['_cum_norm'] = df[mapping['codigo_cum']].apply(normalize_key) if mapping['codigo_cum'] else ""
    df['_nombre_norm'] = df[mapping['nombre']].apply(normalize_desc) if mapping['nombre'] else ""
    df['_pa_norm'] = df[mapping['pa']].apply(normalize_desc) if mapping['pa'] else ""
    df['_tokens'] = df['_nombre_norm'].apply(tokens_from_text)
    df['_cantidad_parsed'] = df[mapping['cantidad']].apply(parse_decimal) if mapping['cantidad'] else None
    df['_cantidad_solicitada_parsed'] = df[mapping['cantidad_solicitada']].apply(parse_decimal) if mapping['cantidad_solicitada'] else None
    df['_tipo_entrega_raw'] = df[mapping['tipo']].fillna("").astype(str) if mapping['tipo'] else ""
    return df, mapping

# ---------------- Classify Row ----------------
def classify_and_accumulate_row(row, tol=TOLERANCE_RELATIVE):
    """
    Returns dict with decimals: delivered_total, delivered_partial, pending, label
    """
    d = row.get('_cantidad_parsed')
    r = row.get('_cantidad_solicitada_parsed')
    tipo = str(row.get('_tipo_entrega_raw','') or '').strip().lower()
    delivered = Decimal(d) if d is not None else None
    requested = Decimal(r) if r is not None else None

    delivered_total = Decimal('0')
    delivered_partial = Decimal('0')
    pending = Decimal('0')
    label = 'unknown'

    # textual priority
    if 'total' in tipo or 'complet' in tipo or 'entrega completa' in tipo:
        # STRICT mode: only use delivered. No imputation from requested.
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

# ---------------- MATCHING LOGIC (Inverse: Find Target for Source) ----------------

def score_source_against_target(src_row, target_idx, target_data):
    """
    Scoring logic adapted: calculates how well `src_row` matches `target_data`.
    target_data is a dict/row from the targets dataframe.
    """
    t_desc_norm = target_data['desc_norm']
    t_pa_norm = target_data['pa_norm']
    t_tokens = target_data['tokens']
    t_codigo_neg = target_data['codigo_neg']
    t_cum = target_data['cum']

    s_desc_norm = src_row.get('_nombre_norm','')
    s_pa_norm = src_row.get('_pa_norm','')
    s_codigo_neg = src_row.get('_codigo_neg_norm','')
    s_cum = src_row.get('_cum_norm','')
    s_tokens = src_row.get('_tokens', [])

    score = 0.0
    reasons = []

    # 1. Exact Codes
    if t_codigo_neg and t_codigo_neg == s_codigo_neg:
        return 1000.0, ['exact_codigo_neg']
    if t_cum and t_cum == s_cum:
        return 900.0, ['exact_cum']

    # 2. Textual Match
    sim_jaccard = jaccard_similarity(t_tokens, s_tokens)
    sim_seq_ratio = seq_ratio(t_desc_norm, s_desc_norm)
    sim_token_sort = token_sort_ratio(t_desc_norm, s_desc_norm)

    score_hybrid = (0.4 * sim_token_sort) + (0.3 * sim_seq_ratio) + (0.3 * sim_jaccard)
    score = score_hybrid * 100

    reasons.append(f'hybrid:{score_hybrid:.3f}')

    if t_desc_norm and s_desc_norm and t_desc_norm == s_desc_norm:
        score += WEIGHTS['desc_exact']
    elif t_desc_norm and s_desc_norm and (t_desc_norm in s_desc_norm or s_desc_norm in t_desc_norm):
        score += WEIGHTS['desc_substring']

    # Principle Active
    if t_pa_norm and s_pa_norm and t_pa_norm == s_pa_norm:
        score += WEIGHTS['same_cum'] # reusing weight name

    return score, reasons

def find_best_target_for_row(row_idx, row, targets_df, token_index_targets):
    """
    Finds the single best target for a given source row.
    """
    # 1. Try Lookup by Code
    s_codigo_neg = row.get('_codigo_neg_norm')
    if s_codigo_neg:
        # Assuming targets_df has indexed lookups passed somehow,
        # or we just scan (slow) or use a pre-built dict.
        # For performance, we should rely on pre-built indices passed in `token_index_targets`.
        matches = token_index_targets['by_codigo_neg'].get(s_codigo_neg, [])
        if matches:
            return matches[0], 1000.0, ['exact_codigo_neg']

    s_cum = row.get('_cum_norm')
    if s_cum:
        matches = token_index_targets['by_cum'].get(s_cum, [])
        if matches:
            return matches[0], 900.0, ['exact_cum']

    # 2. Candidate Selection by Tokens
    s_tokens = row.get('_tokens', [])
    if not s_tokens:
        return None, 0.0, []

    candidates = Counter()
    for tk in s_tokens:
        for tidx in token_index_targets['by_token'].get(tk, []):
            candidates[tidx] += 1

    if not candidates:
        return None, 0.0, []

    # Take top K candidates
    top_candidates = [x[0] for x in candidates.most_common(TOKEN_POOL_TOPK)]

    # 3. Score Candidates
    best_score = -1.0
    best_tidx = None
    best_reasons = []

    for tidx in top_candidates:
        # Retrieve pre-processed target data from dict to avoid DF lookup overhead
        t_data = token_index_targets['data'][tidx]

        # Calculate score
        score, reasons = score_source_against_target(row, tidx, t_data)

        if score > best_score:
            best_score = score
            best_tidx = tidx
            best_reasons = reasons

    return best_tidx, best_score, best_reasons

# ---------------- Processing Worker ----------------

def process_chunk_of_source(chunk_df, targets_df, token_index_targets, mode, strict_cfg, relaxed_cfg):
    """
    Process a chunk of source rows.
    Returns a list of results: (row_idx, assigned_target_idx, score, reasons, classification_dict)
    """
    results = []
    cfg = strict_cfg if mode == 'strict' else relaxed_cfg
    min_score = cfg['MIN_ACCEPT_SCORE']

    for idx, row in chunk_df.iterrows():
        best_tidx, best_score, best_reasons = find_best_target_for_row(idx, row, targets_df, token_index_targets)

        assigned = None
        if best_tidx is not None and best_score >= min_score:
            assigned = best_tidx

        # Special case: if mode is relaxed, check relaxed threshold
        if assigned is None and mode == 'relaxed' and best_tidx is not None:
             if best_score >= relaxed_cfg['MIN_ACCEPT_SCORE']:
                 assigned = best_tidx

        if assigned is not None:
            classif = classify_and_accumulate_row(row)
            results.append({
                'source_idx': idx,
                'target_idx': assigned,
                'score': best_score,
                'reasons': best_reasons,
                'classif': classif
            })
    return results

# ---------------- Main Aggregation Logic ----------------

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

def aggregate(event_files, target_df=None, mode='strict', out_csv='entregas_por_meses_resumen_tipos.csv', audit_csv_prefix=None):

    # 1. Prepare Targets
    if target_df is None:
        logging.info("Generando targets dinámicamente (scan de archivos)...")
        uniq_set = set()

        for f in (event_files or []):
            try:
                tmp = read_csv_robust(Path(f))
                _, m = preprocess_df(tmp) # Just to get column mapping
                # We need simple normalization for dedupe
                tmp['_n'] = tmp[m['nombre']].apply(normalize_desc)
                tmp['_p'] = tmp[m['pa']].apply(normalize_desc)
                for _, r in tmp.iterrows():
                    uniq_set.add((r['_n'], r['_p']))
            except Exception:
                pass
        target_list = [{'DESCRIPCION': u[0], 'PRINCIPIO_ACTIVO': u[1]} for u in uniq_set]
        target_df = pd.DataFrame(target_list)

    # Pre-process Targets ONE TIME
    logging.info(f"Preprocesando {len(target_df)} targets...")
    t_mapping = detect_columns(target_df)

    # Apply normalization to targets
    target_df['_nombre_norm'] = target_df[t_mapping['nombre']].apply(normalize_desc) if t_mapping['nombre'] else ""
    target_df['_pa_norm'] = target_df[t_mapping['pa']].apply(normalize_desc) if t_mapping['pa'] else ""
    target_df['_codigo_neg_norm'] = target_df[t_mapping['codigo_neg']].apply(normalize_key) if t_mapping['codigo_neg'] else ""
    target_df['_cum_norm'] = target_df[t_mapping['codigo_cum']].apply(normalize_key) if t_mapping['codigo_cum'] else ""
    target_df['_tokens'] = target_df['_nombre_norm'].apply(tokens_from_text)

    # Build Target Index for fast lookup
    token_index_targets = {
        'by_token': defaultdict(list),
        'by_codigo_neg': defaultdict(list),
        'by_cum': defaultdict(list),
        'data': {} # Map idx -> pre-computed data dict
    }

    for idx, row in target_df.iterrows():
        token_index_targets['data'][idx] = {
            'desc_norm': row['_nombre_norm'],
            'pa_norm': row['_pa_norm'],
            'codigo_neg': row['_codigo_neg_norm'],
            'cum': row['_cum_norm'],
            'tokens': row['_tokens']
        }

        if row['_codigo_neg_norm']:
            token_index_targets['by_codigo_neg'][row['_codigo_neg_norm']].append(idx)
        if row['_cum_norm']:
            token_index_targets['by_cum'][row['_cum_norm']].append(idx)

        for tk in row['_tokens']:
            token_index_targets['by_token'][tk].append(idx)

    # Structure to hold results: target_idx -> column_name -> Decimal value
    # Initialize with all targets
    agg_results = defaultdict(lambda: defaultdict(Decimal))

    # Audit list
    audit_rows = []

    # 2. Process Files
    all_source_files = []
    
    # Event files
    for f in (event_files or []):
        p = Path(f)
        if not p.exists(): continue
        mn, label = detect_month_from_filename(p.name)
        if not label: label = p.stem
        all_source_files.append({
            'path': p,
            'type': 'evento',
            'label': label,
            'col_prefix': ''
        })

    # Processing Loop
    for src_info in all_source_files:
        p = src_info['path']
        label = src_info['label']
        logging.info(f"Procesando archivo: {p.name} ({label})")

        df_raw = read_csv_robust(p)

        # DEDUPLICATION: Drop duplicate rows from source to prevent counting the same delivery twice
        initial_len = len(df_raw)
        df_raw = df_raw.drop_duplicates()
        dedup_len = len(df_raw)
        if initial_len != dedup_len:
            logging.warning(f"Eliminadas {initial_len - dedup_len} filas duplicadas en {p.name}")

        df, _ = preprocess_df(df_raw)

        # Parallel Processing of Source Rows
        n_cores = psutil.cpu_count(logical=True)
        # Split source into chunks
        chunk_size = int(np.ceil(len(df) / n_cores)) if len(df) > 0 else 1
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        file_results = []

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(process_chunk_of_source, chunk, target_df, token_index_targets, mode, STRICT, RELAXED)
                for chunk in chunks
            ]

            for future in tqdm(as_completed(futures), total=len(chunks), desc=f"Matching {label}"):
                try:
                    res = future.result()
                    file_results.extend(res)
                except Exception as e:
                    logging.error(f"Error en chunk: {e}")

        # Aggregating results for this file
        count_matched = 0
        for res in file_results:
            tidx = res['target_idx']
            classif = res['classif']

            prefix = src_info['col_prefix']

            # For totals accumulation
            agg_results[tidx][f'{prefix}total_entregado_total'] += classif['delivered_total']
            agg_results[tidx][f'{prefix}total_entregado_parcial'] += classif['delivered_partial']
            agg_results[tidx][f'{prefix}total_pendiente'] += classif['pending']

            col_base = f"{prefix}cantidad"

            agg_results[tidx][f'{col_base}_total_{label}'] += classif['delivered_total']
            agg_results[tidx][f'{col_base}_parcial_{label}'] += classif['delivered_partial']
            agg_results[tidx][f'{col_base}_pendiente_{label}'] += classif['pending']

            count_matched += 1

            audit_rows.append({
                'source_file': p.name,
                'source_idx': res['source_idx'],
                'target_idx': tidx,
                'match_score': res['score'],
                'match_reasons': str(res['reasons']),
                'delivered_total': classif['delivered_total'],
                'delivered_partial': classif['delivered_partial'],
                'pending': classif['pending'],
                'label': classif['label']
            })

        logging.info(f"  > Match rate: {count_matched}/{len(df)} rows assigned.")

    # 3. Build Output DataFrame
    logging.info("Construyendo reporte final...")
    
    final_rows = []
    for idx, row in target_df.iterrows():
        out = row.to_dict()
        # cleanup internal cols
        for k in list(out.keys()):
            if k.startswith('_'): del out[k]

        # Add aggregated data
        if idx in agg_results:
            data = agg_results[idx]
            for k, v in data.items():
                out[k] = str(v) if v != 0 else ''

        final_rows.append(out)

    df_out = pd.DataFrame(final_rows)
    
    # Ensure specific columns exist for total accumulation if not present
    totals_cols = ['total_entregado_total', 'total_entregado_parcial', 'total_pendiente']
    for c in totals_cols:
        if c not in df_out.columns:
            df_out[c] = '0'

    # Saving
    df_out.to_csv(out_csv, index=False, sep=';', encoding='utf-8-sig')
    logging.info(f"Guardado resumen: {out_csv}")

    if audit_csv_prefix:
        audit_path = f"{audit_csv_prefix}.csv"
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False, sep=';', encoding='utf-8-sig')
        logging.info(f"Guardado auditoría: {audit_path}")

    return df_out, pd.DataFrame(audit_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cruce entregas (Row-based Matching).")
    parser.add_argument("--targets", help="Ruta a target.csv", default=None)
    parser.add_argument("--event_files", nargs='+', help="Archivos evento", required=False)
    parser.add_argument("--out", help="Output CSV", default="entregas_por_meses_resumen_tipos.csv")
    parser.add_argument("--audit_prefix", help="Audit Prefix", default=None)
    parser.add_argument("--mode", choices=['strict','relaxed'], default='strict')
    parser.add_argument("--tolerance_pct", type=float, default=1.0)
    parser.add_argument("--loglevel", default="INFO")

    if argv is not None:
        parsed = parser.parse_args(argv)
    else:
        parsed, _ = parser.parse_known_args()

    if not parsed.event_files:
        from glob import glob
        candidates = []
        candidates += glob('*evento*.csv') + glob('*Evento*.csv') + glob('*EVENTO*.csv')
        candidates += glob('*abril*.csv') + glob('*mayo*.csv') + glob('*202*.csv')
        seen = set(); candidates_filtered = []
        for c in candidates:
            if c not in seen:
                seen.add(c); candidates_filtered.append(c)
        parsed.event_files = candidates_filtered

    setup_logging(getattr(logging, parsed.loglevel.upper(), logging.INFO))
    global TOLERANCE_RELATIVE
    TOLERANCE_RELATIVE = Decimal(str(parsed.tolerance_pct/100))

    target_df = None
    if parsed.targets:
        target_df = read_csv_robust(Path(parsed.targets))

    aggregate(parsed.event_files, target_df, parsed.mode, parsed.out, parsed.audit_prefix)

if __name__ == "__main__":
    main()
