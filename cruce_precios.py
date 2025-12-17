import pandas as pd
import csv
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict

# ---------------- CONFIG ----------------
TARGETS_FILE = "targets.csv"
COSTOS_FILE  = "costos.csv"
OUTPUT_FILE  = "cruce_precios.csv"
PREVIEW_ROWS = 10

# Umbral de similitud para fallback (si no hay coincidencias exactas)
SIMILARITY_FALLBACK = 0.88

# Encodings a probar y delimitadores
ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

# Regex para detectar precio al final de una cadena (ej: " $360", "$216,795", " 216.795")
_REGEX_PRECIO_FINAL = re.compile(r'[\s\u00A0\-–—:]*\$?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*$')

# ---------------- UTIL ----------------
def detectar_delimitador(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
            if not sample.strip():
                return ","
            dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
            return dialect.delimiter
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as f:
                sample = f.read(4096)
                if not sample.strip():
                    return ","
                dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
                return dialect.delimiter
        except Exception:
            return ","

def leer_csv_con_intentos(ruta, delimitador):
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding=enc, sep=delimitador)
            print(f"Archivo '{ruta}' leído con encoding: {enc} (sep='{delimitador}')")
            return df
        except Exception:
            continue
    try:
        df = pd.read_csv(ruta, dtype=str, keep_default_na=False, sep=delimitador)
        print(f"Archivo '{ruta}' leído con encoding por defecto (sep='{delimitador}')")
        return df
    except Exception as e:
        raise RuntimeError(f"No se pudo leer '{ruta}': {e}")

def remove_zero_width(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r'[\u200B-\u200F\u202A-\u202E\uFEFF]', '', s)

def normalize_nfkc(s: str) -> str:
    return unicodedata.normalize('NFKC', s)

def collapse_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s)

def strip_trailing_price(s: str) -> str:
    if s is None:
        return ""
    s0 = str(s).replace('\ufeff','').replace('\u00A0',' ')
    m = _REGEX_PRECIO_FINAL.search(s0)
    if m:
        # si ocupa toda la cadena, no eliminarla
        if m.start() == 0:
            return s0
        return s0[:m.start()].rstrip()
    return s0

def key_trim(s: str) -> str:
    if s is None:
        return ""
    s0 = str(s).replace('\ufeff','').replace('\u00A0',' ')
    return s0.strip()

def key_casefold(s: str) -> str:
    return key_trim(s).casefold()

def key_robust(s: str) -> str:
    if s is None:
        return ""
    s0 = str(s).replace('\ufeff','').replace('\u00A0',' ')
    s0 = remove_zero_width(s0)
    s0 = normalize_nfkc(s0)
    s0 = collapse_spaces(s0).strip()
    try:
        return s0.casefold()
    except Exception:
        return s0.lower()

def simil(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()

# ---------------- Construir índices desde costos ----------------
def construir_indices(costos_df, col_desc, col_prec):
    """
    Devuelve:
      - list_costs: lista de dicts con info original para fallback (índice fijo)
      - maps: diccionario de mapas para búsquedas rápidas:
            exact -> lista de indices
            trim -> ...
            casefold -> ...
            stripprice_trim -> ...
            robust -> ...
            stripprice_robust -> ...
    """
    list_costs = []  # cada item: {'desc': original_desc, 'precio': precio, 'idx': i}
    maps = {
        'exact': defaultdict(list),
        'trim': defaultdict(list),
        'casefold': defaultdict(list),
        'strip_trim': defaultdict(list),
        'robust': defaultdict(list),
        'strip_robust': defaultdict(list)
    }

    for i, row in costos_df.iterrows():
        desc = row.get(col_desc, "") or ""
        precio = row.get(col_prec, "") or ""
        list_costs.append({'desc': desc, 'precio': precio, 'idx': i})

        maps['exact'][desc].append(i)
        maps['trim'][key_trim(desc)].append(i)
        maps['casefold'][key_casefold(desc)].append(i)
        maps['strip_trim'][ key_trim(strip_trailing_price(desc)) ].append(i)
        maps['robust'][ key_robust(desc) ].append(i)
        maps['strip_robust'][ key_robust(strip_trailing_price(desc)) ].append(i)

    return list_costs, maps

# ---------------- Elegir mejor candidato entre indices (si hay varios) ----------------
def elegir_mejor_por_sim(list_costs, candidatos_indices, target_desc):
    """
    Dado un conjunto de índices candidatos, elige el que tenga mayor similitud
    respecto del target usando key_robust para comparar. Devuelve el índice elegido.
    """
    if not candidatos_indices:
        return None
    best_idx = candidatos_indices[0]
    best_score = -1.0
    target_k = key_robust(target_desc)
    for ci in candidatos_indices:
        cand_desc = list_costs[ci]['desc']
        score = simil(target_k, key_robust(cand_desc))
        if score > best_score:
            best_score = score
            best_idx = ci
    return best_idx, best_score

# ---------------- Matching por orden de certeza ----------------
def match_target(desc_target, list_costs, maps):
    # claves del target en diferentes transformaciones
    k_exact = desc_target
    k_trim = key_trim(desc_target)
    k_casefold = key_casefold(desc_target)
    k_strip_trim = key_trim(strip_trailing_price(desc_target))
    k_robust = key_robust(desc_target)
    k_strip_robust = key_robust(strip_trailing_price(desc_target))

    # 1) exact -> exact
    if k_exact in maps['exact']:
        idxs = maps['exact'][k_exact]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'exact->exact', 1.0

    # 2) trim_target -> exact_cost (target had extra edges)
    if k_trim in maps['exact']:
        idxs = maps['exact'][k_trim]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'trim_target->exact_cost', 1.0

    # 3) stripPrice_target -> exact_cost
    if k_strip_trim in maps['exact']:
        idxs = maps['exact'][k_strip_trim]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'stripPrice_target->exact_cost', 1.0

    # 4) exact_target -> trim_cost
    if k_exact in maps['trim']:
        idxs = maps['trim'][k_exact]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'exact_target->trim_cost', 1.0

    # 5) trim_target -> trim_cost
    if k_trim in maps['trim']:
        idxs = maps['trim'][k_trim]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'trim_target->trim_cost', 1.0

    # 6) casefold (insensible a mayúsculas)
    if k_casefold in maps['casefold']:
        idxs = maps['casefold'][k_casefold]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'casefold_target->casefold_cost', 1.0

    # 7) robust matches
    if k_robust in maps['robust']:
        idxs = maps['robust'][k_robust]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'robust_target->robust_cost', 1.0

    # 8) stripPrice comparisons against robust cost
    if k_strip_robust in maps['robust']:
        idxs = maps['robust'][k_strip_robust]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'stripPrice_target->robust_cost', 1.0

    # 9) strip_robust maps directly
    if k_strip_robust in maps['strip_robust']:
        idxs = maps['strip_robust'][k_strip_robust]
        chosen_idx, score = elegir_mejor_por_sim(list_costs, idxs, desc_target)
        return list_costs[chosen_idx]['precio'], 'stripPrice_target->stripRobust_cost', 1.0

    # 10) fallback por similitud: comparar target robust vs todos los costos robust y elegir el mejor si supera umbral
    best_global_idx = None
    best_global_score = 0.0
    target_k = k_robust
    for i, item in enumerate(list_costs):
        cand_k = key_robust(item['desc'])
        score = simil(target_k, cand_k)
        if score > best_global_score:
            best_global_score = score
            best_global_idx = i

    if best_global_idx is not None and best_global_score >= SIMILARITY_FALLBACK:
        return list_costs[best_global_idx]['precio'], f'fallback_sim_{best_global_score:.2f}', best_global_score

    # no encontrado razonable
    return "", "no_encontrado", 0.0

# ---------------- PROGRAMA PRINCIPAL ----------------
def main():
    delim_costos = detectar_delimitador(COSTOS_FILE)
    delim_targets = detectar_delimitador(TARGETS_FILE)

    try:
        costos = leer_csv_con_intentos(COSTOS_FILE, delim_costos)
        targets = leer_csv_con_intentos(TARGETS_FILE, delim_targets)
    except Exception as e:
        print("Error leyendo archivos:", e)
        sys.exit(1)

    # columnas esperadas (fallback)
    col_cost_desc = 'descripcion'
    col_cost_prec = 'costo'
    col_target_desc = 'descripcion'

    if col_cost_desc not in costos.columns:
        if len(costos.columns) > 0:
            costos.rename(columns={costos.columns[0]: col_cost_desc}, inplace=True)
            print(f"Advertencia: renombrada primera columna de '{COSTOS_FILE}' a '{col_cost_desc}'")
    if col_cost_prec not in costos.columns:
        if len(costos.columns) > 1:
            costos.rename(columns={costos.columns[1]: col_cost_prec}, inplace=True)
            print(f"Advertencia: renombrada segunda columna de '{COSTOS_FILE}' a '{col_cost_prec}'")
        else:
            costos[col_cost_prec] = ""
            print(f"Advertencia: no se encontró columna de precio en '{COSTOS_FILE}'. Se creó '{col_cost_prec}' vacía.")
    if col_target_desc not in targets.columns:
        if len(targets.columns) > 0:
            targets.rename(columns={targets.columns[0]: col_target_desc}, inplace=True)
            print(f"Advertencia: renombrada primera columna de '{TARGETS_FILE}' a '{col_target_desc}'")

    # construir índices
    list_costs, maps = construir_indices(costos, col_cost_desc, col_cost_prec)

    precios_encontrados = []
    metodos = []
    similitudes = []
    matched_cost_desc = []

    found_count = 0
    not_found_list = []

    for idx, row in targets.iterrows():
        desc_t = row.get(col_target_desc, "") or ""
        precio, metodo, sim = match_target(desc_t, list_costs, maps)

        precios_encontrados.append(precio)
        metodos.append(metodo)
        similitudes.append(sim)

        if sim and float(sim) > 0:
            if sim >= 1.0:
                found_count += 1
            elif sim >= SIMILARITY_FALLBACK:
                found_count += 1
        if metodo == "no_encontrado":
            not_found_list.append((idx, desc_t))

    resultado = targets.copy()
    resultado['PrecioEncontrado'] = precios_encontrados
    resultado['MetodoCoincidencia'] = metodos
    resultado['Similitud'] = similitudes

    # guardar resultado con mismo separador que targets
    try:
        resultado.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', sep=delim_targets)
        print(f"\nResultado guardado en: {OUTPUT_FILE} (sep='{delim_targets}')")
    except Exception as e:
        print(f"Error guardando con sep='{delim_targets}': {e}. Intentando con coma.")
        resultado.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', sep=",")
        print(f"Resultado guardado en: {OUTPUT_FILE} (coma)")

    total = len(targets)
    not_found = total - found_count
    print(f"\nResumen: {found_count}/{total} encontrados ({not_found} no encontrados).")
    print("\nVista previa:")
    print(resultado.head(PREVIEW_ROWS))

    # Mostrar diagnóstico: para los primeros no encontrados, listar top-3 candidatos por similitud
    if not_found > 0:
        print("\n=== DIAGNÓSTICO: primeros no encontrados y top-3 candidatos por similitud ===")
        costos_descs = [it['desc'] for it in list_costs]
        max_show = 10
        for i, (idx, desc_t) in enumerate(not_found_list[:max_show]):
            print(f"\nTarget idx {idx} repr: {repr(desc_t)}")
            krob = key_robust(desc_t)
            scored = []
            for it in list_costs:
                score = simil(krob, key_robust(it['desc']))
                scored.append((score, it['desc'], it['precio']))
            scored.sort(reverse=True, key=lambda x: x[0])
            for s, d, p in scored[:3]:
                print(f"  sim={s:.3f} repr_cost={repr(d)} precio='{p}'")
    print("\nProceso completado.")

if __name__ == "__main__":
    main()
