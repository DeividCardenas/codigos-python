import re
import unicodedata

# Diccionario de Formas Farmacéuticas (Normalización)
# Mapea variaciones a un término estándar.
FORMS_MAP = {
    # TABLETAS
    'TABLETAS': 'TABLETA', 'TABLETA': 'TABLETA', 'TAB': 'TABLETA',
    'TAB REC': 'TABLETA', 'TAB LIB PROL': 'TABLETA', 'TBS': 'TABLETA', 'TABS': 'TABLETA',
    'COMP': 'TABLETA', 'COM': 'TABLETA', 'GRAGEA': 'TABLETA', 'GRAGEAS': 'TABLETA',

    # CAPSULAS
    'CAPSULA': 'CAPSULA', 'CAPSULAS': 'CAPSULA', 'CAP': 'CAPSULA', 'CAPS': 'CAPSULA',
    'CAP BLAND': 'CAPSULA', 'BLANDA': 'CAPSULA', 'CAPSULA BLANDA': 'CAPSULA',
    'CAP DURA': 'CAPSULA', 'CAPSULA DURA': 'CAPSULA', 'CP': 'CAPSULA',

    # LIQUIDOS
    'SOLUCION': 'SOLUCION', 'SOL': 'SOLUCION', 'LIQ': 'SOLUCION', 'SLN': 'SOLUCION',
    'GOTAS': 'SOLUCION', 'GTS': 'SOLUCION',
    'JARABE': 'SOLUCION', 'JBE': 'SOLUCION', 'JRB': 'SOLUCION', 'SYR': 'SOLUCION',
    'SUSPENSION': 'SOLUCION', 'SUSP': 'SOLUCION', 'SUS': 'SOLUCION',
    'ELIXIR': 'SOLUCION',

    # INYECTABLES
    'INYECTABLE': 'INYECTABLE', 'INY': 'INYECTABLE',
    'AMPOLLA': 'INYECTABLE', 'AMP': 'INYECTABLE',
    'VIAL': 'INYECTABLE', 'JERINGA': 'INYECTABLE', 'LAPICERO': 'INYECTABLE',
    'PRELLENADA': 'INYECTABLE', 'CARTUCHO': 'INYECTABLE',

    # TOPICOS
    'CREMA': 'CREMA', 'CRM': 'CREMA',
    'UNGUENTO': 'UNGUENTO', 'UNG': 'UNGUENTO', 'POMADA': 'UNGUENTO', 'POM': 'UNGUENTO',
    'GEL': 'GEL',
    'LOCION': 'LOCION',
    'PARCHE': 'PARCHE',

    # UNIDADES / EMPAQUE
    'FRASCO': 'UNIDAD', 'FCO': 'UNIDAD',
    'SOBRE': 'UNIDAD', 'SOB': 'UNIDAD',
    'BOLSA': 'UNIDAD', 'UND': 'UNIDAD',
    'CAJA': 'UNIDAD', 'KIT': 'UNIDAD',
    'UNIDAD': 'UNIDAD',
    'POLVO': 'POLVO', 'GRANULOS': 'POLVO',
    'AEROSOL': 'AEROSOL', 'INHALADOR': 'AEROSOL', 'SPRAY': 'AEROSOL', 'PUFF': 'AEROSOL',
    'OVULO': 'OVULO', 'SUPOSITORIO': 'SUPOSITORIO'
}

# Palabras a ignorar al extraer componentes (Ruido)
IGNORED_WORDS = {
    'DE', 'PARA', 'CON', 'Y', 'EN', 'LA', 'EL', 'LOS', 'LAS', 'DEL', 'AL',
    'MG', 'MCG', 'G', 'ML', 'UI', 'IU', 'GR', '%',
    'CAJA', 'FRASCO', 'BLISTER', 'TUBO', 'SOBRE', 'X', 'POR', 'CANTIDAD',
    'RECUBIERTA', 'LIBERACION', 'PROLONGADA', 'RETARD', 'FORTE', 'PLUS',
    'ORAL', 'TOPICO', 'VAGINAL', 'NASAL', 'OFTALMICO', 'RECTAL',
    'ADULTO', 'PEDIATRICO', 'INFANTIL', 'NEONATAL',
    'ACIDO', # A veces es parte del nombre (Acido Acetilsalicilico), pero a veces ruido.
             # Estrategia: Si "ACIDO" esta solo, ignorar. Si es "ACIDO X", X es el componente.
             # Por simplicidad en normalización A+B, "ACIDO" suele estorbar el orden.
             # Mejor estrategia: Dejar ACIDO adherido si es posible, o ignorarlo para el sort.
             # User said: "Ordena alfabéticamente los componentes". "Acido Valproico" vs "Valproico Acido".
             # Si quitamos ACIDO, queda VALPROICO en ambos. Es mas seguro quitarlo para matching robusto.
    'SODIO', 'CALCIO', 'POTASIO', 'MAGNESIO', 'ZINC', 'HIERRO', # Sales comunes
    'CLORHIDRATO', 'BROMURO', 'SULFATO', 'FUMARATO', 'MALEATO', 'SUCCINATO', 'VALERATO', 'PROPIONATO'
}

def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_text(text):
    if not text: return ""
    text = str(text).upper()
    text = remove_accents(text)
    # Reemplazar caracteres especiales por espacio, excepto + y . y %
    text = re.sub(r'[^A-Z0-9\+\.\%\s]', ' ', text)
    # Reemplazar multiples espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_medication(text):
    """
    Analiza una cadena de medicamento y retorna sus partes estructuradas.
    Retorna diccionario:
        - components: list (sorted)
        - dosages: set (floats)
        - form: str (normalized form)
        - original_clean: str
    """
    clean_text = normalize_text(text)

    # 1. Extraer Forma Farmacéutica
    # Buscamos la coincidencia más larga posible en el mapa
    found_form = "DESCONOCIDA"

    # Tokenizar para buscar formas
    tokens = clean_text.split()

    # Heurística simple: Buscar tokens que coincidan con FORMS_MAP
    # Si encontramos "CAPSULA" y "BLANDA", priorizar "CAPSULA BLANDA"
    # Iterar sobre el texto buscando frases del mapa

    # Ordenar claves de FORMS_MAP por longitud descendente para match greedy
    sorted_forms = sorted(FORMS_MAP.keys(), key=len, reverse=True)

    temp_text_for_form = clean_text

    for key in sorted_forms:
        # Regex con word boundaries
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, temp_text_for_form):
            found_form = FORMS_MAP[key]
            # Eliminar la forma del texto para no confundirla con componentes
            temp_text_for_form = re.sub(pattern, ' ', temp_text_for_form)
            break

    # 2. Extraer Dosis (Numeros + Unidades)
    dosages = set()
    # Regex para numeros con unidad
    # Capture: (Num) (Unit)
    dose_pattern = r'\b(\d+(?:[\.,]\d+)?)\s*(MG|MCG|G|UI|IU|GR|ML|%)\b'

    matches = re.findall(dose_pattern, temp_text_for_form)

    # Remover dosis del texto
    temp_text_no_doses = re.sub(dose_pattern, ' ', temp_text_for_form)
    # Tambien remover numeros sueltos que puedan haber quedado (ej. "500" sin mg si estaba mal escrito, o ruido)
    # Pero cuidado con nombres quimicos con numeros (ej. B12).
    # Mejor solo remover lo que matcheamos como dosis.

    for val_str, unit in matches:
        val_norm = val_str.replace(',', '.')
        try:
            val = float(val_norm)
            # Normalizar unidades si es necesario (ej % -> mg/ml?)
            # El usuario dijo: "Extrae los valores numéricos y compáralos uno a uno".
            # No pidió conversión explícita, pero para match "500mg" vs "500", necesitamos el numero.
            dosages.add(val)
        except:
            pass

    # 3. Extraer Componentes
    # Lo que queda en temp_text_no_doses son los componentes + ruido

    # Dividir por '+' si existe
    raw_components = []
    if '+' in temp_text_no_doses:
        parts = temp_text_no_doses.split('+')
        raw_components = parts
    else:
        # Si no hay +, asumimos un solo bloque, PERO puede haber varios componentes sin + (mala practica input)
        # Asumiremos el string entero como un componente por ahora, limpiando ruido.
        raw_components = [temp_text_no_doses]

    final_components = []

    for part in raw_components:
        # Limpiar palabras ignoradas dentro del componente
        part_tokens = part.split()
        clean_tokens = []
        for t in part_tokens:
            if t not in IGNORED_WORDS and not t.isdigit() and len(t) > 1:
                clean_tokens.append(t)

        if clean_tokens:
            comp_str = " ".join(clean_tokens)
            final_components.append(comp_str)

    # Ordenar alfabéticamente (Regla de Oro: A+B = B+A)
    final_components.sort()

    # Si dosages esta vacio, intentar buscar numeros sueltos en el original?
    # A veces "Acetaminofen 500" sin unidad.
    if not dosages:
         nums = re.findall(r'\b(\d+(?:[\.,]\d+)?)\b', temp_text_for_form)
         for n in nums:
             try:
                 dosages.add(float(n.replace(',', '.')))
             except:
                 pass

    return {
        'components': tuple(final_components), # Tuple for hashing
        'dosages': tuple(sorted(list(dosages))), # Sorted tuple for hashing/comparison
        'form': found_form,
        'original_clean': clean_text
    }
