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
    'JBE': 'SOLUCION', 'JRB': 'SOLUCION', 'SYR': 'SOLUCION', 'JARABE': 'SOLUCION', # Jarabe -> Solucion
    'SUSP': 'SUSPENSION', 'SUS': 'SUSPENSION',
    'SOL': 'SOLUCION', 'SLN': 'SOLUCION', 'LIQ': 'LIQUIDO',
    'INY': 'INYECTABLE', 'INYEC': 'INYECTABLE',
    'AMP': 'AMPOLLA', 'AMPO': 'AMPOLLA', 'VIA': 'VIAL',
    'GTS': 'SOLUCION', 'GOTA': 'SOLUCION', 'GOTAS': 'SOLUCION', # Gotas -> Solucion
    'UNG': 'UNGUENTO', 'POM': 'POMADA', 'CRM': 'CREMA', 'CREM': 'CREMA',
    'SUP': 'SUPOSITORIO', 'OV': 'OVULO',
    'INH': 'INHALADOR', 'AER': 'AEROSOL', 'PFF': 'PUFF',
    'SOB': 'SOBRE', 'PLV': 'SOLUCION', 'POL': 'SOLUCION', 'POLVO': 'SOLUCION', # Polvo -> Solucion (simplicacion)
    'MCG': 'MCG', 'UG': 'MCG', 'UI': 'UI', 'IU': 'UI',
    'MG': 'MG', 'G': 'G', 'ML': 'ML', 'L': 'L',
    'GRAG': 'GRAGEA', 'GRAGEAS': 'GRAGEAS',
    'LEUPROLIDE': 'LEUPRORELINA', 'JERINGA': 'INYECTABLE', 'PRELLENADA': '',
    'IMPLANTE': 'INYECTABLE', 'VIAL': '', 'AMPOLLA': '', 'AMPO': '', 'AMP': '',
    'ESTERIL': '', 'RECONSTITUIR': '', 'PARA': '', 'DE': '',
    'ADAPALENE': 'ADAPALENO',
    'URSODEOXICOLICO': 'URSODESOXICOLICO'
}

# Palabras de parada para validación estricta (no cuentan como coincidencias clave)
# Se incluyen sales comunes y prefijos genéricos para forzar match en el principio activo real
STOP_WORDS = {
    'TABLETA', 'TABLETAS', 'CAPSULA', 'CAPSULAS', 'SOLUCION', 'SUSPENSION',
    'INYECTABLE', 'JARABE', 'CREMA', 'GEL', 'UNGUENTO', 'POMADA',
    'MG', 'MCG', 'G', 'ML', 'L', 'UI', 'IU', 'POR', 'DE', 'PARA', 'Y', 'CON',
    'CAJA', 'FRASCO', 'AMPOLLA', 'VIAL', 'SOBRE', 'TUBO', 'BLISTER',
    'ORAL', 'TOPICO', 'VAGINAL', 'NASAL', 'OFTALMICO', 'RECUBIERTA',
    'LIBERACION', 'PROLONGADA', 'RETARD', 'FORTE', 'PLUS', 'DIA', 'NOCHE',
    'ADULTO', 'PEDIATRICO', 'INFANTIL', 'NEONATAL',
    'ACIDO', 'SODIO', 'CALCIO', 'CARBONATO', 'SULFATO', 'CLORURO',
    'NITRATO', 'OXIDO', 'POTASIO', 'MAGNESIO', 'HIDROXIDO', 'FOSFATO',
    'ACETATO', 'ZINC', 'HIERRO', 'FUMARATO', 'MALEATO', 'SUCCINATO',
    'VALERATO', 'PROPIONATO', 'DIPROPIONATO', 'BETAMETASONA', # Generic component
    'HIDROCLORURO', 'CLORHIDRATO', 'BROMURO', 'TARTRATO', 'CITRATO'
}

def simple_soundex(token):
    """
    Implementación simple de Soundex para Español/Farmacéutico.
    Retorna un código fonético.
    """
    if not token: return ""
    token = token.upper()

    # 1. Primera letra
    first_char = token[0]

    # 2. Mapeos
    # B, F, P, V -> 1
    # C, G, J, K, Q, S, X, Z -> 2
    # D, T -> 3
    # L -> 4
    # M, N -> 5
    # R -> 6
    # Vocales, H, W, Y -> 0 (se ignoran)

    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    code = first_char
    prev_digit = mapping.get(first_char, '0')

    for char in token[1:]:
        digit = mapping.get(char, '0')
        if digit != '0' and digit != prev_digit:
            code += digit
            prev_digit = digit

    # Pad or trim to 4 chars
    return (code + "0000")[:4]

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

    # 3.5 Normalizar decimales (coma a punto)
    text = text.replace(',', '.')

    # 4. Separar números de letras (500MG -> 500 MG) y porcentajes
    # Ex: 500MG -> 500 MG, 10ML -> 10 ML, 0.5% -> 0.5 %
    text = re.sub(r'(\d)\s*([A-Z%]+)', r'\1 \2', text)
    text = re.sub(r'([A-Z%]+)\s*(\d)', r'\1 \2', text) # T3 -> T 3 (raro pero pasa)

    # 5. Normalizar espacios y caracteres
    text = re.sub(r'[^A-Z0-9\s\.\%]', ' ', text) # Dejar puntos para decimales si los hay, aunque parseamos nums aparte

    # 6. Expandir abreviaturas
    tokens = text.split()
    expanded = []
    for t in tokens:
        val = PHARMA_ABBREVIATIONS.get(t, t)
        if val:
            expanded.append(val)

    return " ".join(expanded).strip()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def check_key_token_match(src_text, tgt_text):
    """
    Verifica que al menos un token 'clave' (no stopword, no numero) coincida.
    Ayuda a evitar falsos positivos como 'Acido Ursodeoxicolico' vs 'Acido Tioctico'.
    """
    def get_tokens(t):
        # Tokenizar, quitar numeros y stopwords
        raw_tokens = t.upper().replace('.', ' ').split()
        clean = set()
        for tok in raw_tokens:
            if tok.isdigit(): continue
            if re.match(r'^\d+[\.,]?\d*$', tok): continue # float like
            if len(tok) < 3: continue # muy cortos
            if tok in STOP_WORDS: continue
            clean.add(tok)
        return clean

    s_toks = get_tokens(src_text)
    t_toks = get_tokens(tgt_text)

    # Si no quedan tokens (ej: solo dosis), asumimos match valido si paso las otras reglas
    if not s_toks or not t_toks:
        return True

    # Verificar PRIMER TOKEN SIGNIFICATIVO (First Token Check)
    # Ordenamos tokens? No, tomamos el orden original pero filtrado.
    # Requerimos que tokens de entrada esten ordenados para esto.
    # Mejor: Tomamos la lista original y filtramos stop words en orden.

    def get_ordered_tokens(t):
        raw = t.upper().replace('.', ' ').split()
        clean = []
        for tok in raw:
            if tok.isdigit(): continue
            if re.match(r'^\d+[\.,]?\d*$', tok): continue
            if len(tok) < 3: continue
            if tok in STOP_WORDS: continue
            clean.append(tok)
        return clean

    s_list = get_ordered_tokens(src_text)
    t_list = get_ordered_tokens(tgt_text)

    if not s_list or not t_list:
        # Fallback a intersección simple si se eliminó todo
        return bool(set(s_list) & set(t_list)) if s_list or t_list else True

    # REGLA ESTRICTA: El primer token significativo debe coincidir (fuzzy > 85 O Fonetico)
    # Esto previene "ACIDO URSODEOXICOLICO" (1er=URSODEOXICOLICO) vs "ACIDO TIOCTICO" (1er=TIOCTICO)
    # (Asumiendo que ACIDO esta en STOP_WORDS)

    s_first = s_list[0]
    t_first = t_list[0]

    # 1. Match Exacto
    if s_first == t_first:
        return True

    # 2. Match Fuzzy Alto
    if _HAS_RAPIDFUZZ:
        if fuzz.ratio(s_first, t_first) > 85:
            return True

    # 3. Match Fonético (Soundex)
    # Solo si la longitud es decente para evitar falsos positivos en cortos (ej: GEL vs GEL)
    if len(s_first) >= 4 and len(t_first) >= 4:
        if simple_soundex(s_first) == simple_soundex(t_first):
            return True

    return False

def extract_features(text):
    """
    Extrae características clave:
    - Set de números (dosis).
    - Set de unidades/tokens clave.
    """
    nums = set()

    # Solo buscar números asociados a unidades de potencia/concentración
    # Regex: numero seguido (opcionalmente con espacio) de unidad
    # Unidades: MG, MCG, G, UI, IU, %, GR (evitar ML si no es parte de algo mas complejo, pero aqui simplificamos)
    # Tambien extraemos numeros sueltos SI el texto es muy corto? No.

    # Patrones específicos
    # 1. X MG/ML o X %
    # 2. X MG
    # 3. X MCG

    # Estrategia: Buscar todos los pares (Numero, Unidad)
    # Si la unidad es MG, MCG, G, UI, IU -> Guardar Numero
    # Si la unidad es ML -> Ignorar (a menos que sea concentracion? dificil saber)
    # Si la unidad es % -> Guardar Numero Y convertir a mg/ml (x10) para ophthalmic drops

    # Primero normalizamos espacios alrededor de unidades para regex simple
    # clean_text ya separa numeros de letras (500 MG)

    # Buscar: \bNUMBER\s+(MG|MCG|G|UI|IU|GR|%)\b
    # Nota: % no tiene word boundary \b al final necesariamente
    matches = re.findall(r'\b(\d+(?:[\.,]\d+)?)\s+(?:(MG|MCG|G|UI|IU|GR)\b|(%))', text)
    for m in matches:
        val_str = m[0]
        unit = m[1] if m[1] else m[2]
        val_norm = val_str.replace(',', '.')
        try:
            val_float = float(val_norm)
            nums.add(val_float)

            # Conversion Percentage -> mg/ml (assuming w/v)
            if unit == '%':
                # 1% = 10 mg/ml
                nums.add(val_float * 10.0)
        except:
            pass

    # 2. Detección de Concentración Implícita (X MG / Y ML)
    # Aunque '/' se elimina en limpieza, queda "X MG Y ML"
    matches_conc = re.findall(r'\b(\d+(?:[\.,]\d+)?)\s+MG\s+(\d+(?:[\.,]\d+)?)\s+ML\b', text)
    for m in matches_conc:
        try:
            mg = float(m[0].replace(',', '.'))
            ml = float(m[1].replace(',', '.'))
            if ml > 0:
                # Calcular concentración mg/ml
                nums.add(mg / ml)
        except:
            pass

    # Caso especial: Si no encontramos nada, quiza buscar numeros muy grandes (>25) que suelen ser mg?
    # No, peligroso (podria ser cantidad de tabletas).

    return {'nums': nums}

def check_feature_compatibility(src_feats, tgt_feats):
    """
    Devuelve un factor de penalización (0.0 a 1.0).
    1.0 = Compatible.
    0.1 = Incompatible (dosis diferente).
    """
    s_nums = src_feats['nums']
    t_nums = tgt_feats['nums']

    # Si alguno no tiene numeros extraidos (porque no tenia unidad explicita), no penalizar.
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
        self.target_df['_clean_no_nums'] = self.target_df['_clean'].apply(remove_numbers)
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
            # Ajustar n_neighbors dinámicamente si hay pocos targets (caso Secondary)
            n_samples = self.target_df.shape[0]
            k_neighbors = min(5, n_samples)
            if k_neighbors < 1: k_neighbors = 1

            self.nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', n_jobs=-1)
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
            clean_desc_no_nums = remove_numbers(clean_desc)
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
                final_score = 0

                # Lógica Avanzada: Si hay compatibilidad numérica perfecta, ignorar números en texto
                if compatibility == 1.0:
                    tgt_clean_no_nums = tgt_row['_clean_no_nums']
                    score_no_nums = 0
                    score_with_nums = 0

                    if _HAS_RAPIDFUZZ:
                        score_no_nums = fuzz.token_set_ratio(clean_desc_no_nums, tgt_clean_no_nums)
                        score_with_nums = fuzz.token_set_ratio(clean_desc, tgt_clean)
                    else:
                        score_no_nums = 50 if clean_desc_no_nums in tgt_clean_no_nums else 0
                        score_with_nums = 50 if clean_desc in tgt_clean else 0

                    # Usar el mejor de los dos mundos
                    final_score = max(score_no_nums, score_with_nums)

                    # LOGICA DE RESCATE (RESCUE LOGIC)
                    # Si el score está entre 60 y 75, pero la compatibilidad de dosis es PERFECTA (1.0)
                    # Y ademas pasa la validacion de Tokens Clave (Fonetica/Fuzzy)
                    # Le subimos artificialmente el score para que pase el corte RELAXED (75.0)
                    if 60.0 <= final_score < 75.0:
                        if check_key_token_match(clean_desc_no_nums, tgt_clean_no_nums):
                            # Bonus variable para asegurar cruce
                            # Si es >70, +5 basta. Si es 60, necesitamos +15
                            final_score = 75.1
                            best_method = "RESCUED_PHONETIC_60+"

                else:
                    # Compatibilidad dudosa o mala, usar texto completo y penalizar
                    if _HAS_RAPIDFUZZ:
                        base_score = fuzz.token_set_ratio(clean_desc, tgt_clean)
                    else:
                        base_score = 50 if clean_desc in tgt_clean else 0
                    final_score = base_score * compatibility

                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx

            results[local_idx] = (best_idx, best_score, best_method)

        return results

# ---------------- PROCESAMIENTO PRINCIPAL ----------------

def process_file(filepath, matcher, matcher_secondary, rescued_matches_list, output_unmatched_list):
    print(f"\nProcesando archivo: {filepath.name}")
    try:
        df = read_csv_robust(filepath)
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return {}, {}, Decimal('0'), Decimal('0') # source_tracker, results, total, matched

    cols = detect_columns(df)
    if not cols['cantidad']:
        print("   -> ERROR: No se encontró columna de cantidad. Saltando.")
        return {}, {}, Decimal('0'), Decimal('0')

    col_nom = cols['nombre']
    col_cod = cols['codigo']
    col_cant = cols['cantidad']

    total_qty = Decimal('0')
    matched_qty = Decimal('0')
    local_results = defaultdict(Decimal) # (TargetID, SourceType) -> Qty

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

        # 1. Match Batch PRIMARIO
        match_results = matcher.match_batch(batch_descs, batch_codes)

        # 2. Match Batch SECUNDARIO (Cascade)
        # Identificar indices que fallaron en primario
        secondary_indices = []
        secondary_inputs_desc = []
        secondary_inputs_code = []

        # Pre-scan results to find what needs secondary
        for j, (tidx, score, method) in enumerate(match_results):
             is_match = False
             if tidx is not None:
                if score >= THRESHOLD_CONFIDENT or score >= THRESHOLD_RELAXED:
                    is_match = True

             if not is_match and matcher_secondary:
                 secondary_indices.append(j)
                 secondary_inputs_desc.append(batch_descs[j])
                 secondary_inputs_code.append(batch_codes[j])

        secondary_results = []
        if secondary_indices and matcher_secondary:
             secondary_results = matcher_secondary.match_batch(secondary_inputs_desc, secondary_inputs_code)

        # Procesar resultados del lote (Merge Primary + Secondary)
        sec_ptr = 0
        for j, (tidx, score, method) in enumerate(match_results):
            qty_val = parse_decimal(batch_qties[j])
            total_qty += qty_val

            final_tidx = tidx
            final_score = score
            final_method = method
            source_type = "PRIMARY"

            # Check if Primary was success
            is_match = False
            if final_tidx is not None:
                if final_score >= THRESHOLD_CONFIDENT or final_score >= THRESHOLD_RELAXED:
                    is_match = True

            # If not matched in primary, check secondary
            if not is_match and matcher_secondary and sec_ptr < len(secondary_results):
                # Check secondary result
                s_tidx, s_score, s_method = secondary_results[sec_ptr]
                sec_ptr += 1

                # Check threshold for secondary
                is_sec_match = False
                if s_tidx is not None:
                     if s_score >= THRESHOLD_CONFIDENT or s_score >= THRESHOLD_RELAXED:
                        is_sec_match = True

                if is_sec_match:
                    final_tidx = s_tidx
                    final_score = s_score
                    final_method = s_method
                    source_type = "SECONDARY"
                    is_match = True

            if is_match:
                matched_qty += qty_val
                # Key for aggregation needs to be unique across primary/secondary or handled downstream
                # Since Target IDs might overlap (0, 1, 2...), we need to disambiguate or ensure IDs are unique.
                # Here we will store tuple (ID, SourceType)
                local_results[(final_tidx, source_type)] += qty_val

                # Capture rescued matches for audit (Primary or Secondary)
                if "RESCUED" in final_method and rescued_matches_list is not None:
                     # Fetch description from correct dataframe
                     if source_type == "PRIMARY":
                         tgt_desc = matcher.target_df.iloc[final_tidx][matcher.col_desc]
                     else:
                         tgt_desc = matcher_secondary.target_df.iloc[final_tidx][matcher_secondary.col_desc]

                     rescued_matches_list.append({
                        'Archivo': filepath.name,
                        'Descripcion_Original': batch_descs[j],
                        'Match_Target': tgt_desc,
                        'Score': final_score,
                        'Method': final_method + f" ({source_type})"
                     })
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
    print(f"   -> Total: {total_qty:,.0f} | Cruzado: {matched_qty:,.0f} | Cobertura: {pct:.2f}%")
    
    # Track sources count just for summary
    return local_results, total_qty, matched_qty

def main():
    parser = argparse.ArgumentParser(description="Cruce Farmacéutico Avanzado")
    parser.add_argument("--targets", default="Targets.csv", help="Archivo maestro de productos (PRIMARIO)")
    parser.add_argument("--targets_secondary", default="Targets_Secondary.csv", help="Archivo maestro de productos (SECUNDARIO)")
    parser.add_argument("--files", nargs='+', help="Archivos de movimiento (meses)")
    parser.add_argument("--batch_size", type=int, default=5000, help="Tamaño del lote de procesamiento")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    # 1. Cargar Targets Primarios
    target_path = Path(args.targets)
    if not target_path.exists():
        print("Error: No existe Targets.csv")
        return

    print(f"Cargando maestro PRIMARIO: {target_path}")
    target_df = read_csv_robust(target_path)
    matcher = SmartMatcher(target_df)

    # 2. Cargar Targets Secundarios
    matcher_secondary = None
    target_sec_path = Path(args.targets_secondary)
    if target_sec_path.exists():
        print(f"Cargando maestro SECUNDARIO: {target_sec_path}")
        try:
            target_sec_df = read_csv_robust(target_sec_path)
            if not target_sec_df.empty:
                matcher_secondary = SmartMatcher(target_sec_df)
                print(" -> Matcher Secundario Activo")
        except Exception as e:
            print(f"Advertencia: Error cargando targets secundarios: {e}")
    else:
        print(" -> No se encontró targets secundarios (Targets_Secondary.csv), modo simple.")

    # 3. Buscar archivos si no se pasan
    files = args.files
    if not files:
        import glob
        files = glob.glob("*Evento*.csv") + glob.glob('*evento*.csv') + glob.glob('*EVENTO*.csv')
        files += glob.glob('*abril*.csv') + glob.glob('*mayo*.csv') + glob.glob('*202*.csv')
        # Eliminar duplicados manteniendo orden
        files = sorted(list(set(files)))
        print(f"Archivos detectados automáticamente: {len(files)}")

    # 4. Procesar
    audit_unmatched = []
    rescued_matches = []

    # Map ID -> Info (Need separate maps for Primary and Secondary)
    # Aggregated Pivot: Key = (ID, SourceType)

    target_info_primary = {}
    for idx, row in target_df.iterrows():
        desc = row[matcher.col_desc] if matcher.col_desc else ""
        code = row[matcher.col_code] if matcher.col_code else ""
        target_info_primary[idx] = {'desc': desc, 'code': code, 'source': 'PRIMARY'}

    target_info_secondary = {}
    if matcher_secondary:
        for idx, row in matcher_secondary.target_df.iterrows():
            desc = row[matcher_secondary.col_desc] if matcher_secondary.col_desc else ""
            code = row[matcher_secondary.col_code] if matcher_secondary.col_code else ""
            target_info_secondary[idx] = {'desc': desc, 'code': code, 'source': 'SECONDARY'}

    pivot_data = defaultdict(dict) # (ID, SourceType) -> {Month: Qty}

    global_total = Decimal('0')
    global_matched = Decimal('0')

    for fname in files:
        fpath = Path(fname)
        results, f_total, f_matched = process_file(fpath, matcher, matcher_secondary, rescued_matches, audit_unmatched)

        global_total += f_total
        global_matched += f_matched

        col_name = fpath.stem
        # results keys are (tidx, source_type)
        for (tidx, src_type), qty in results.items():
            pivot_data[(tidx, src_type)][col_name] = qty

    # 4. Generar Reportes
    print("\nGenerando reporte final...")

    # Resumen de Rescates
    if rescued_matches:
        print("\n" + "="*40)
        print(f"RESUMEN DE RESCATES (Score Original 60-75% + Phonetic) - Total: {len(rescued_matches)}")
        print("Muestra de los primeros 20 (Ordenados por Score Asc):")
        rescued_matches.sort(key=lambda x: x['Score'])
        for r in rescued_matches[:20]:
            print(f"   [{r['Score']:.1f}|{r.get('Method','')}] {r['Descripcion_Original'][:30]}... -> {r['Match_Target'][:30]}...")
        print("="*40 + "\n")

    # Reporte No Encontrados
    if audit_unmatched:
        df_unmatched = pd.DataFrame(audit_unmatched)
        unmatched_file = f"no_encontrados_avanzado_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        df_unmatched.to_csv(unmatched_file, index=False, sep=';', encoding='utf-8-sig')
        print(f"Reporte de NO cruzados guardado en: {unmatched_file}")

        # Generar Reporte de Top Faltantes para accionabilidad
        try:
            # Agrupar por Descripcion Original para ver qué falta crear en Targets
            missing_agg = df_unmatched.groupby('Descripcion_Original').agg({
                'Cantidad': 'sum',
                'Codigo_Original': 'first' # Un ejemplo de codigo
            }).reset_index()

            # Ordenar por cantidad descendente
            missing_agg = missing_agg.sort_values('Cantidad', ascending=False).head(50)

            missing_file = f"top_missing_targets_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            missing_agg.to_csv(missing_file, index=False, sep=';', encoding='utf-8-sig')

            print("\n" + "="*60)
            print("TOP 50 PRODUCTOS FALTANTES (Candidatos para agregar a Master):")
            print(f"Archivo guardado en: {missing_file}")
            print("-" * 60)
            # Imprimir bonito
            print(f"{'CANTIDAD':>10} | {'CODIGO':<10} | {'DESCRIPCION'}")
            for _, row in missing_agg.iterrows():
                print(f"{row['Cantidad']:10.0f} | {str(row['Codigo_Original'])[:10]:<10} | {row['Descripcion_Original'][:60]}")
            print("="*60 + "\n")

        except Exception as e:
            print(f"No se pudo generar reporte de faltantes: {e}")

    # Reporte Cruzado (Pivot)
    final_rows = []
    
    # Obtener todos los nombres de meses/archivos encontrados
    all_months = sorted([Path(f).stem for f in files])

    for (tidx, src_type), month_dict in pivot_data.items():
        info = None
        if src_type == 'PRIMARY':
            info = target_info_primary.get(tidx, {'desc':'?', 'code':'?'})
        else:
            info = target_info_secondary.get(tidx, {'desc':'?', 'code':'?'})

        row = {
            'FUENTE_MATCH': src_type,
            'DESCRIPCION_TARGET': info['desc'],
            'CODIGO_TARGET': info['code']
        }
        total_row = Decimal('0')
        for m in all_months:
            qty = month_dict.get(m, Decimal('0'))
            row[m] = str(qty).replace('.', ',') if qty > 0 else '0'
            total_row += qty

        row['TOTAL_ACUMULADO'] = str(total_row).replace('.', ',')
        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)
    # Ordenar columnas
    cols_order = ['FUENTE_MATCH', 'DESCRIPCION_TARGET', 'CODIGO_TARGET'] + all_months + ['TOTAL_ACUMULADO']
    # Asegurar que solo usemos columnas que existen
    cols_order = [c for c in cols_order if c in df_final.columns]
    
    if not df_final.empty:
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