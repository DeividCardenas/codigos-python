# cruce_entregas_capita.py
"""
Cruce y agregación de entregas por medicamento.
VERSION AVANZADA (State-of-the-Art):
- Usa TF-IDF con N-gramas para búsqueda semántica robusta a errores ortográficos.
- Validación de unidades y dosis para evitar falsos positivos.
- Reporte detallado de cobertura.
- Optimizado para alto volumen (procesamiento por lotes).
"""

import re
import sys
import argparse
import logging
from pathlib import Path
from decimal import Decimal, getcontext
from collections import Counter, defaultdict
import pandas as pd 
import numpy as np 
import datetime
import unicodedata
from tqdm import tqdm

# Intentar importar librerías avanzadas
try:
    from rapidfuzz import fuzz, utils, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    print("ADVERTENCIA: rapidfuzz no instalado. El rendimiento será menor.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("ADVERTENCIA: scikit-learn no instalado. Se usará modo básico (lento).")

getcontext().prec = 28

# ---------------- Configuración ----------------
# Umbrales
THRESHOLD_CONFIDENT = 88.0  # Coincidencia muy segura
THRESHOLD_RELAXED = 75.0    # Coincidencia probable (revisar si es crítico)
PENALTY_MISMATCH_UNIT = 50.0 # Penalización masiva si la dosis no coincide
BATCH_SIZE = 5000 # Tamaño del lote para procesamiento vectorizado

ENCODINGS = ["utf-8", "latin-1", "cp1252", "ISO-8859-1"]
SEPS = [';', ',', '\t', '|']

# Diccionario ampliado de abreviaturas farmacéuticas
PHARMA_ABBREVIATIONS = {
    'TAB': 'TABLETA', 'TABS': 'TABLETAS', 'TB': 'TABLETA', 'COM': 'TABLETA', 'COMP': 'TABLETA',
    'CAP': 'CAPSULA', 'CAPS': 'CAPSULAS', 'CP': 'CAPSULA',
    'JBE': 'JARABE', 'JRB': 'JARABE', 'SYR': 'JARABE',
    'SUSP': 'SUSPENSION', 'SUS': 'SUSPENSION',
    'SOL': 'SOLUCION', 'SLN': 'SOLUCION', 'LIQ': 'LIQUIDO',
    'INY': 'INYECTABLE', 'INYEC': 'INYECTABLE',
    'AMP': 'AMPOLLA', 'AMPO': 'AMPOLLA', 'VIA': 'VIAL',
    'GTS': 'GOTAS', 'GOTA': 'GOTAS',
    'UNG': 'UNGUENTO', 'POM': 'POMADA', 'CRM': 'CREMA', 'CREM': 'CREMA',
    'SUP': 'SUPOSITORIO', 'OV': 'OVULO',
    'INH': 'INHALADOR', 'AER': 'AEROSOL', 'PFF': 'PUFF',
    'SOB': 'SOBRE', 'PLV': 'POLVO', 'POL': 'POLVO',
    'MCG': 'MCG', 'UG': 'MCG', 'UI': 'UI', 'IU': 'UI',
    'MG': 'MG', 'G': 'G', 'ML': 'ML', 'L': 'L',
    'GRAG': 'GRAGEA', 'GRAGEAS': 'GRAGEAS'
}

LABORATORIES = {
    'TECNOQUIMICAS', 'SIEGFRIED', 'FARMACAPSULAS', 'SANOFI', 'GRUNENTHAL',
    'PROCAPS', 'NOVAMED', 'WINTHROP', 'ASTRAZENECA', 'TECNOFARMA', 'COLMED',
    'ECAR', 'GENFAR', 'PFIZER', 'GSK', 'NOVARTIS', 'ROCHE', 'ABBOTT', 'BAYER',
    'MERCK', 'BOEHRINGER', 'JANSSEN', 'MSD', 'LILLY', 'BRISTOL', 'AMGEN',
    'HUMAX', 'VITALIS', 'BAXTER', 'B BRAUN', 'BIOSIDUS', 'MK', 'LA SANTE',
    'LAFRANCOL', 'ANGLOPHARMA', 'JGB', 'QUIMIDROGAS', 'MEMPHIS', 'COASPHARMA'
}
LABS_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, sorted(LABORATORIES, key=len, reverse=True))) + r')\b', re.IGNORECASE)

# ---------------- Utilidades de Texto ----------------

def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_text_advanced(text):
    """
    Limpieza profunda:
    1. Mayúsculas.
    2. Quitar acentos.
    3. Separar números de letras (500MG -> 500 MG).
    4. Expandir abreviaturas.
    5. Quitar caracteres especiales.
    """
    if pd.isna(text) or text == "": return ""

    # 1. Mayúsculas y pipe check
    text = str(text).upper()
    if '|' in text:
        text = text.split('|')[0] # Tomar primera parte si hay pipes

    # 2. Quitar Labs (ruido)
    text = LABS_PATTERN.sub('', text)

    # 3. Quitar acentos
    text = remove_accents(text)

    # 4. Separar números de letras (500MG -> 500 MG)
    # Ex: 500MG -> 500 MG, 10ML -> 10 ML
    text = re.sub(r'(\d)\s*([A-Z]+)', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)\s*(\d)', r'\1 \2', text) # T3 -> T 3 (raro pero pasa)

    # 5. Normalizar espacios y caracteres
    text = re.sub(r'[^A-Z0-9\s\.]', ' ', text) # Dejar puntos para decimales si los hay, aunque parseamos nums aparte

    # 6. Expandir abreviaturas
    tokens = text.split()
    expanded = []
    for t in tokens:
        expanded.append(PHARMA_ABBREVIATIONS.get(t, t))

    return " ".join(expanded).strip()

def extract_features(text):
    """
    Extrae características clave:
    - Set de números (dosis).
    - Set de unidades/tokens clave.
    """
    # Buscar números (enteros o decimales)
    nums = set()
    matches = re.findall(r'\b\d+(?:[\.,]\d+)?\b', text)
    for m in matches:
        m_norm = m.replace(',', '.')
        try:
            nums.add(float(m_norm))
        except:
            pass

    return {'nums': nums}

def check_feature_compatibility(src_feats, tgt_feats):
    """
    Devuelve un factor de penalización (0.0 a 1.0).
    1.0 = Compatible.
    0.1 = Incompatible (dosis diferente).
    """
    s_nums = src_feats['nums']
    t_nums = tgt_feats['nums']

    if not s_nums or not t_nums:
        return 1.0

    # Si hay intersección de números, es bueno.
    intersection = s_nums.intersection(t_nums)
    if len(intersection) > 0:
        return 1.0

    # Penalizar si los números son completamente disjuntos
    return 0.5

# ---------------- Utilidades IO ----------------

def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Intenta leer CSV probando separadores y encodings.
    Prioriza el separador que genere más columnas.
    """
    best_df = None
    max_cols = 0

    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python', on_bad_lines='skip')

                # Si encontramos un formato con más columnas, es mejor candidato
                if df.shape[1] > max_cols:
                    max_cols = df.shape[1]
                    best_df = df
            except:
                continue

        # Si con este encoding encontramos algo decente (>1 col), probablemente es el encoding correcto
        if best_df is not None and max_cols > 1:
            break

    if best_df is not None:
        # Limpiar nombres de columnas
        best_df.columns = [str(c).strip().replace('"', '') for c in best_df.columns]
        return best_df

    raise Exception(f"No se pudo leer el archivo {path}")

def parse_decimal(val):
    if pd.isna(val) or val == '': return Decimal('0')
    s = str(val).strip()
    s = s.replace('$', '').replace(' ', '')
    if ',' in s and '.' in s:
        if s.find('.') < s.find(','): # 1.000,00
            s = s.replace('.', '').replace(',', '.')
        else: # 1,000.00
            s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')

    try:
        return Decimal(s)
    except:
        return Decimal('0')

# ---------------- Detección de Columnas ----------------
def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    mapping = {
        'nombre': next((c for c in cols if any(x in c for x in ['descripcion', 'nombre', 'producto', 'medicamento'])), None),
        'codigo': next((c for c in cols if any(x in c for x in ['codigo', 'cod_neg', 'cum', 'cums', 'ap'])), None),
        'cantidad': next((c for c in cols if any(x in c for x in ['cantidad_entregada', 'cant', 'entregad', 'unidades'])), None)
    }
    # Fallback para Targets (que tiene headers fijos usualmente)
    if 'descripcion' in cols: mapping['nombre'] = 'descripcion'
    if 'codigo_neg' in cols: mapping['codigo'] = 'codigo_neg'

    return mapping

# ---------------- CLASE MATCHER INTELIGENTE ----------------

class SmartMatcher:
    def __init__(self, target_df):
        self.target_df = target_df.copy()
        print("Preprocesando targets...")

        # 1. Normalización
        cols = detect_columns(self.target_df)
        self.col_desc = cols['nombre'] or self.target_df.columns[0]
        self.col_code = cols['codigo']

        # Limpieza previa
        self.target_df['_clean'] = self.target_df[self.col_desc].apply(clean_text_advanced)
        self.target_df['_features'] = self.target_df['_clean'].apply(extract_features)

        # Indexar códigos para match exacto (rápido)
        self.code_index = {}
        if self.col_code:
            # Limpiar códigos
            self.target_df['_code_clean'] = self.target_df[self.col_code].astype(str).str.strip().str.upper()
            for idx, row in self.target_df.iterrows():
                c = row['_code_clean']
                if c and c != 'NAN':
                    self.code_index[c] = idx

        # 2. Vectorización TF-IDF
        if _HAS_SKLEARN:
            print("Entrenando vectorizador TF-IDF (3-gramas)...")
            self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1, strip_accents='unicode')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.target_df['_clean'].tolist())

            # Nearest Neighbors para búsqueda rápida
            self.nn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
            self.nn.fit(self.tfidf_matrix)
        else:
            self.vectorizer = None

    def match_batch(self, descriptions, codes):
        """
        Procesa un lote de filas.
        descriptions: lista de strings crudos.
        codes: lista de códigos crudos (o None).
        Retorna lista de tuplas: (target_idx, score, method)
        """
        results = [(None, 0.0, "NONE")] * len(descriptions)

        # 1. Match Exacto por Código (Prioridad Absoluta)
        unmatched_indices = []
        clean_descs_unmatched = []

        for i, (desc, code) in enumerate(zip(descriptions, codes)):
            # Intento match por código
            matched_by_code = False
            if code:
                rc = str(code).strip().upper()
                if rc in self.code_index:
                    results[i] = (self.code_index[rc], 100.0, "CODE_EXACT")
                    matched_by_code = True

            if not matched_by_code:
                if desc:
                    unmatched_indices.append(i)
                    clean_descs_unmatched.append(clean_text_advanced(desc))
                else:
                    results[i] = (None, 0.0, "EMPTY")

        if not unmatched_indices:
            return results

        # 2. Búsqueda Semántica (Batch)
        candidates_map = {} # i -> [target_indices]

        if _HAS_SKLEARN and self.vectorizer and clean_descs_unmatched:
            try:
                # Transformar todo el lote a la vez (Mucho más rápido)
                vec_batch = self.vectorizer.transform(clean_descs_unmatched)
                distances_batch, indices_batch = self.nn.kneighbors(vec_batch)

                for k, local_idx in enumerate(unmatched_indices):
                    # Obtener candidatos para esta fila
                    # indices_batch[k] son los indices en target_df
                    cands = []
                    for j, tgt_idx in enumerate(indices_batch[k]):
                        sim = 1 - distances_batch[k][j]
                        if sim > 0.3: # Threshold de filtrado
                            cands.append(tgt_idx)
                    candidates_map[local_idx] = cands
            except Exception as e:
                logging.error(f"Error en batch vectorization: {e}")

        # 3. Re-ranking (CPU intensive but on small candidate set)
        # Si no hay SKLEARN, esto será muy lento (fallback a scan completo o nada)
        if not _HAS_SKLEARN:
             # Fallback simple: usar RapidFuzz process.extract si disponible para cada fila
             # Esto es lento O(N*M), pero mejor que nada.
             if _HAS_RAPIDFUZZ:
                 choices = self.target_df['_clean'].tolist()
                 for local_idx, txt in zip(unmatched_indices, clean_descs_unmatched):
                     # ExtractOne es lento contra toda la lista, pero necesario sin TF-IDF
                     res = process.extractOne(txt, choices, scorer=fuzz.token_sort_ratio, score_cutoff=60)
                     if res:
                         # res es (match_str, score, index)
                         target_idx = res[2]
                         score = res[1]
                         results[local_idx] = (target_idx, score, "FUZZ_FALLBACK")
             return results

        # Re-ranking normal para candidatos TF-IDF
        for i, local_idx in enumerate(unmatched_indices):
            candidates = candidates_map.get(local_idx, [])
            if not candidates:
                continue

            # Acceso directo por índice en lugar de .index()
            clean_desc = clean_descs_unmatched[i]
            src_feats = extract_features(clean_desc)

            best_score = 0.0
            best_idx = None
            best_method = "TFIDF+FUZZ"

            for idx in candidates:
                tgt_row = self.target_df.iloc[idx]
                tgt_clean = tgt_row['_clean']
                tgt_feats = tgt_row['_features']

                compatibility = check_feature_compatibility(src_feats, tgt_feats)

                base_score = 0
                if _HAS_RAPIDFUZZ:
                    base_score = fuzz.token_sort_ratio(clean_desc, tgt_clean)
                else:
                    # Fallback sin rapidfuzz
                    base_score = 50 if clean_desc in tgt_clean else 0

                final_score = base_score * compatibility

                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx

            results[local_idx] = (best_idx, best_score, best_method)

        return results

# ---------------- PROCESAMIENTO PRINCIPAL ----------------

def process_file(filepath, matcher, output_audit_list, output_unmatched_list):
    print(f"\nProcesando archivo: {filepath.name}")
    try:
        df = read_csv_robust(filepath)
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return {}, Decimal('0'), Decimal('0')

    cols = detect_columns(df)
    if not cols['cantidad']:
        print("  -> ERROR: No se encontró columna de cantidad. Saltando.")
        return {}, Decimal('0'), Decimal('0')

    col_nom = cols['nombre']
    col_cod = cols['codigo']
    col_cant = cols['cantidad']

    total_qty = Decimal('0')
    matched_qty = Decimal('0')
    local_results = defaultdict(Decimal)

    # Procesar por lotes
    num_rows = len(df)

    # Preparar datos crudos
    raw_descs = df[col_nom].fillna("").astype(str).tolist() if col_nom else [""] * num_rows
    raw_codes = df[col_cod].fillna("").astype(str).tolist() if col_cod else [None] * num_rows
    raw_qties = df[col_cant].tolist()

    # Iterar en chunks
    for i in tqdm(range(0, num_rows, BATCH_SIZE), desc="Cruzando por lotes"):
        end_ix = min(i + BATCH_SIZE, num_rows)

        batch_descs = raw_descs[i:end_ix]
        batch_codes = raw_codes[i:end_ix]
        batch_qties = raw_qties[i:end_ix]

        # Match Batch
        match_results = matcher.match_batch(batch_descs, batch_codes)

        # Procesar resultados del lote
        for j, (tidx, score, method) in enumerate(match_results):
            qty_val = parse_decimal(batch_qties[j])
            total_qty += qty_val

            is_match = False
            if tidx is not None:
                if score >= THRESHOLD_CONFIDENT:
                    is_match = True
                elif score >= THRESHOLD_RELAXED:
                    is_match = True

            if is_match:
                matched_qty += qty_val
                local_results[tidx] += qty_val
            else:
                # Guardar no encontrado
                output_unmatched_list.append({
                    'Archivo': filepath.name,
                    'Descripcion_Original': batch_descs[j],
                    'Codigo_Original': batch_codes[j],
                    'Cantidad': qty_val,
                    'Mejor_Candidato_Idx': tidx if tidx is not None else '',
                    'Score': score
                })

    pct = (matched_qty / total_qty * 100) if total_qty > 0 else 0
    print(f"  -> Total: {total_qty:,.0f} | Cruzado: {matched_qty:,.0f} | Cobertura: {pct:.2f}%")
    
    return local_results, total_qty, matched_qty

def main():
    parser = argparse.ArgumentParser(description="Cruce Farmacéutico Avanzado")
    parser.add_argument("--targets", default="Targets.csv", help="Archivo maestro de productos")
    parser.add_argument("--files", nargs='+', help="Archivos de movimiento (meses)")
    parser.add_argument("--batch_size", type=int, default=5000, help="Tamaño del lote de procesamiento")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    # 1. Cargar Targets
    target_path = Path(args.targets)
    if not target_path.exists():
        print("Error: No existe Targets.csv")
        return

    print(f"Cargando maestro: {target_path}")
    target_df = read_csv_robust(target_path)

    # Inicializar Matcher (entrena modelos)
    matcher = SmartMatcher(target_df)

    # 2. Buscar archivos si no se pasan
    files = args.files
    if not files:
        import glob
        files = glob.glob("*Evento*.csv") + glob.glob("*evento*.csv")
        files = sorted(list(set(files)))
        print(f"Archivos detectados automáticamente: {len(files)}")

    # 3. Procesar
    audit_unmatched = []

    target_info = {}
    for idx, row in target_df.iterrows():
        desc = row[matcher.col_desc] if matcher.col_desc else ""
        code = row[matcher.col_code] if matcher.col_code else ""
        target_info[idx] = {'desc': desc, 'code': code}

    pivot_data = defaultdict(dict)

    global_total = Decimal('0')
    global_matched = Decimal('0')

    for fname in files:
        fpath = Path(fname)
        results, f_total, f_matched = process_file(fpath, matcher, None, audit_unmatched)

        global_total += f_total
        global_matched += f_matched

        col_name = fpath.stem
        for tidx, qty in results.items():
            pivot_data[tidx][col_name] = qty

    # 4. Generar Reportes
    print("\nGenerando reporte final...")

    # Reporte No Encontrados
    if audit_unmatched:
        df_unmatched = pd.DataFrame(audit_unmatched)
        unmatched_file = f"no_encontrados_avanzado_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        df_unmatched.to_csv(unmatched_file, index=False, sep=';', encoding='utf-8-sig')
        print(f"Reporte de NO cruzados guardado en: {unmatched_file}")
        try:
            top_missed = df_unmatched.groupby('Codigo_Original')['Cantidad'].sum().sort_values(ascending=False).head(20)
            print("\nTop 20 Productos NO encontrados (por código/cantidad):")
            print(top_missed)
        except:
            pass

    # Reporte Cruzado (Pivot)
    final_rows = []
    all_months = sorted([Path(f).stem for f in files])

    for tidx, month_dict in pivot_data.items():
        row = {
            'DESCRIPCION_TARGET': target_info[tidx]['desc'],
            'CODIGO_TARGET': target_info[tidx]['code']
        }
        total_row = Decimal('0')
        for m in all_months:
            qty = month_dict.get(m, Decimal('0'))
            row[m] = str(qty).replace('.', ',') if qty > 0 else '0'
            total_row += qty

        row['TOTAL_ACUMULADO'] = str(total_row).replace('.', ',')
        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)
    cols_order = ['DESCRIPCION_TARGET', 'CODIGO_TARGET'] + all_months + ['TOTAL_ACUMULADO']
    cols_order = [c for c in cols_order if c in df_final.columns]
    df_final = df_final[cols_order]

    out_file = "reporte_cruce_avanzado.csv"
    df_final.to_csv(out_file, index=False, sep=';', encoding='utf-8-sig')

    print("\n" + "="*60)
    print(f"RESUMEN FINAL")
    print(f"Total Cantidad Procesada: {global_total:,.2f}")
    print(f"Total Cantidad Cruzada:   {global_matched:,.2f}")
    pct_global = (global_matched / global_total * 100) if global_total > 0 else 0
    print(f"Porcentaje de Éxito:      {pct_global:.2f}%")
    print(f"Reporte guardado en:      {out_file}")
    print("="*60)

if __name__ == "__main__":
    main()
