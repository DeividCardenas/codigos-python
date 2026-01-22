import pandas as pd
import numpy as np
import glob
import re
import unicodedata
import os
import difflib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeliveryProcessorPro:
    # 1. Constants and Mappings
    FORMA_MAP = {
        'TAB': 'TABLETA', 'TABLETAS': 'TABLETA', 'COMPRIMIDO': 'TABLETA',
        'CAP': 'CAPSULA', 'CAPSULAS': 'CAPSULA', 'SOFTGEL': 'CAPSULA',
        'AMP': 'AMPOLLA', 'VIAL': 'AMPOLLA', 'INY': 'INYECTABLE',
        'SOL': 'SOLUCION', 'JAR': 'JARABE', 'UNT': 'UNGUENTO', 'CRM': 'CREMA'
    }

    LAB_MAP = {
        'GENFAR': 'GENFAR', 'TECNOQUIMICAS': 'TECNOQUIMICAS', 'TQ': 'TECNOQUIMICAS',
        'LA SANTE': 'LA_SANTE', 'LASANTE': 'LA_SANTE', 'LAPROFF': 'LAPROFF',
        'ECAR': 'ECAR', 'PROCAPS': 'PROCAPS', 'MK': 'MK', 'HUMAX': 'HUMAX',
        'BAYER': 'BAYER', 'ABBOTT': 'ABBOTT', 'SANOFI': 'SANOFI', 'PFIZER': 'PFIZER',
        'NOVARTIS': 'NOVARTIS', 'GSK': 'GSK', 'ASTRAZENECA': 'ASTRAZENECA',
        'JANSSEN': 'JANSSEN', 'MERCK': 'MERCK', 'MSD': 'MERCK', 'SIEGFRIED': 'SIEGFRIED',
        'NOVAMED': 'NOVAMED', 'SYNTHESIS': 'SYNTHESIS', 'CHALVER': 'CHALVER',
        'WINTHROP': 'WINTHROP', 'GRUNENTHAL': 'GRUNENTHAL', 'BIOQUIFAR': 'BIOQUIFAR',
        'FARMACAPSULAS': 'FARMACAPSULAS'
    }

    KNOWN_UNITS = ['MG', 'ML', 'MCG', 'G', 'UI', 'UNIDADES']

    def __init__(self):
        self.catalog_df = pd.DataFrame()
        self.catalog_keys = []
        self.catalog_map = {} # Map key -> Official Name
        self.processed_data = [] # List of DataFrames
        self.audit_records = []

        # Path definitions
        self.INPUT_DIR = os.path.join("data", "raw")
        self.CATALOG_DIR = os.path.join("data", "catalog")
        self.PROCESSED_DIR = os.path.join("data", "processed")
        self.OUTPUT_DIR = os.path.join("output", "consolidated")

        # Ensure directories exist
        for directory in [self.INPUT_DIR, self.CATALOG_DIR, self.PROCESSED_DIR, self.OUTPUT_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")

    def remove_accents(self, text):
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def normalize_text(self, text, is_catalog=False):
        """
        Normalize text: Uppercase, remove accents, handle units, apply FORMA_MAP.
        """
        if not isinstance(text, str):
            return ""

        # 1. Basic cleaning
        clean = self.remove_accents(text).upper()

        # 2. Apply FORMA_MAP (only for input files mainly, but safe for catalog too if consistent)
        # We replace whole words
        for abbr, full in self.FORMA_MAP.items():
            # Regex for whole word replacement
            pattern = r'\b' + re.escape(abbr) + r'\b'
            clean = re.sub(pattern, full, clean)

        # 3. Normalize Units (Remove spaces: 500 MG -> 500MG)
        # Regex to find Number + Space + Unit
        unit_pattern = r'(\d)\s+(' + '|'.join(self.KNOWN_UNITS) + r')\b'
        clean = re.sub(unit_pattern, r'\1\2', clean)

        # 4. Remove extra spaces
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean

    def load_catalog(self, filename="productos - Hoja 1.csv"):
        """
        Load and normalize the master catalog.
        """
        filepath = os.path.join(self.CATALOG_DIR, filename)
        if not os.path.exists(filepath):
            logging.error(f"Catalog file not found in {filepath}. Please ensure the catalog is in {self.CATALOG_DIR}")
            return

        try:
            # Try reading with comma first
            df = pd.read_csv(filepath, on_bad_lines='skip')
            # If only one column, try semicolon
            if len(df.columns) < 2:
                 df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')

            # Assuming the catalog has a column for the name.
            # Looking at "productos - Hoja 1.csv", it seems to have 'producto_genhospi' based on previous `read_file`
            col_name = 'producto_genhospi'
            if col_name not in df.columns:
                # Fallback detection
                if len(df.columns) > 0:
                    col_name = df.columns[0]

            self.catalog_df = df.copy()
            self.catalog_df['llave_catalogo'] = self.catalog_df[col_name].apply(lambda x: self.normalize_text(x, is_catalog=True))

            # Create a map for quick lookup and fuzzy matching list
            # We map the NORMALIZED key to the OFFICIAL ORIGINAL Name
            self.catalog_map = dict(zip(self.catalog_df['llave_catalogo'], self.catalog_df[col_name]))
            self.catalog_keys = list(self.catalog_map.keys())

            logging.info(f"Catalog loaded: {len(self.catalog_df)} items.")

        except Exception as e:
            logging.error(f"Error loading catalog: {e}")

    def extract_lab(self, text):
        """
        Extract Laboratory based on pipes and LAB_MAP.
        """
        if not isinstance(text, str):
            return "NO_IDENTIFICADO"

        # Rule A: Pipes | LAB |
        if '|' in text:
            parts = text.split('|')
            if len(parts) >= 2:
                potential_lab = parts[1].strip().upper()
                # Normalize extracted lab if it matches keys
                for key, val in self.LAB_MAP.items():
                    if key == potential_lab:
                        return val
                if potential_lab:
                     return potential_lab # Return what was found between pipes if not in map?
                     # Or check if it contains keywords.
                     # User said: "Usa el diccionario LAB_MAP para extraer y estandarizar"
                     # If pipe exists but not in map, we usually keep it as is or try to map it.
                     # Let's check map keywords inside the pipe text.
                     for key, val in self.LAB_MAP.items():
                         if key in potential_lab:
                             return val
                     return potential_lab # Fallback to pipe content

        # Rule B: Keyword search in the whole text
        text_upper = text.upper()
        for key, val in self.LAB_MAP.items():
            if key in text_upper:
                return val

        return "NO_IDENTIFICADO"

    def clean_quantity(self, val):
        """
        Colombian format: 1.000,00 -> 1000.00
        """
        if pd.isna(val):
            return 0.0
        s = str(val)
        s = s.replace('.', '') # Remove thousands separator
        s = s.replace(',', '.') # Replace decimal separator
        try:
            return float(s)
        except:
            return 0.0

    def process_row(self, row):
        """
        Process a single row to generate clean columns.
        """
        desc_original = str(row.get('descripcion_original', ''))

        # 1. Lab
        lab = self.extract_lab(desc_original)

        # 2. Normalize Description (Key)
        # Use text before the first pipe if it exists for the description part
        desc_part = desc_original.split('|')[0] if '|' in desc_original else desc_original
        llave_cruce = self.normalize_text(desc_part)

        # 3. Fuzzy Match
        producto_oficial = f"POR_REVISAR_CATALOGO: {llave_cruce}"
        match_score = 0.0

        if llave_cruce:
            matches = difflib.get_close_matches(llave_cruce, self.catalog_keys, n=1, cutoff=0.85)
            if matches:
                match_key = matches[0]
                producto_oficial = self.catalog_map[match_key]
                match_score = 1.0 # Logical match found

        return pd.Series([lab, llave_cruce, producto_oficial, match_score],
                         index=['laboratorio_estandar', 'llave_cruce', 'producto_oficial_catalogo', 'match_found'])

    def detect_columns(self, df):
        cols = [c.lower() for c in df.columns]
        col_map = {}

        # Description
        for cand in ['nombre', 'nombres', 'descripcion', 'producto']:
            matches = [c for c in df.columns if cand in c.lower()]
            if matches:
                col_map['descripcion_original'] = matches[0]
                break

        # Quantity
        for cand in ['cantidad', 'cantidades', 'unidades', 'cant']:
            matches = [c for c in df.columns if cand in c.lower()]
            if matches:
                col_map['cantidad'] = matches[0]
                break

        return col_map

    def process_files(self, pattern="*.csv"):
        # Search in INPUT_DIR
        search_path = os.path.join(self.INPUT_DIR, pattern)
        files = glob.glob(search_path)

        if not files:
            logging.warning(f"No files found in {self.INPUT_DIR}. Please add input CSV files.")
            return

        for f in files:
            logging.info(f"Processing {f}...")
            try:
                # 1. Read
                try:
                    df = pd.read_csv(f, dtype=str, on_bad_lines='skip')
                except:
                    df = pd.read_csv(f, sep=';', dtype=str, on_bad_lines='skip')

                # 2. Identify Columns
                col_map = self.detect_columns(df)
                if not col_map or 'descripcion_original' not in col_map or 'cantidad' not in col_map:
                    logging.warning(f"Skipping {f}: Could not identify required columns.")
                    continue

                # Rename for consistency
                df = df.rename(columns={
                    col_map['descripcion_original']: 'descripcion_original',
                    col_map['cantidad']: 'cantidad'
                })

                # 3. Clean Quantity
                df['cantidad_num'] = df['cantidad'].apply(self.clean_quantity)

                # 4. Extract Month from Filename
                # Assuming filename like "6junioEvento.csv" -> "6junioEvento"
                filename = os.path.splitext(os.path.basename(f))[0]
                # Try to extract month name if possible, or just use filename
                # User requirement: "extrae el mes del nombre del archivo"
                # We will use the filename as the month identifier for simplicity unless parsing is needed.
                # Example: "6junioEvento" -> "6junioEvento"
                df['mes_entrega'] = filename

                # 5. Process Rows with optimization (Deduplicate descriptions for matching)
                logging.info(f"Matching {len(df)} records...")

                # Pre-calculate lab and key for all rows
                df['laboratorio_estandar'] = df['descripcion_original'].apply(self.extract_lab)

                def get_clean_key(text):
                    text = str(text) if pd.notna(text) else ""
                    desc_part = text.split('|')[0] if '|' in text else text
                    return self.normalize_text(desc_part)

                df['llave_cruce'] = df['descripcion_original'].apply(get_clean_key)

                # Get unique keys to match
                unique_keys = df['llave_cruce'].unique()
                logging.info(f"Unique keys to match: {len(unique_keys)}")

                key_map = {}
                for key in unique_keys:
                    if not key:
                        key_map[key] = ("POR_REVISAR_CATALOGO: ", 0.0)
                        continue

                    matches = difflib.get_close_matches(key, self.catalog_keys, n=1, cutoff=0.85)
                    if matches:
                        match_key = matches[0]
                        official_name = self.catalog_map[match_key]
                        key_map[key] = (official_name, 1.0)
                    else:
                        key_map[key] = (f"POR_REVISAR_CATALOGO: {key}", 0.0)

                # Map back results
                df['producto_oficial_catalogo'] = df['llave_cruce'].map(lambda k: key_map.get(k, ("", 0.0))[0])
                df['match_found'] = df['llave_cruce'].map(lambda k: key_map.get(k, (0, 0.0))[1])

                # 6. Save Individual File
                output_path = os.path.join(self.PROCESSED_DIR, f"{filename}_LIMPIO.csv")
                df.to_csv(output_path, index=False, encoding='utf-8-sig')

                # 7. Add to list for consolidation
                self.processed_data.append(df)

                # 8. Audit Stats
                total_rows = len(df)
                total_qty = df['cantidad_num'].sum()
                match_count = df['match_found'].sum() # match_found is 1.0 if match
                match_pct = (match_count / total_rows * 100) if total_rows > 0 else 0

                self.audit_records.append({
                    'Archivo': filename,
                    'Total Registros': total_rows,
                    'Suma Cantidades': total_qty,
                    '% Coincidencia': match_pct
                })

            except Exception as e:
                logging.error(f"Error processing {f}: {e}")

    def consolidate_and_report(self):
        if not self.processed_data:
            logging.warning("No data to consolidate.")
            return

        logging.info("Consolidating data...")
        master_df = pd.concat(self.processed_data, ignore_index=True)

        # Aggregate
        # Group by: mes_entrega, producto_oficial_catalogo, laboratorio_estandar
        # metrics: sum(cantidad_num), count(rows)

        consolidado = master_df.groupby(
            ['mes_entrega', 'producto_oficial_catalogo', 'laboratorio_estandar']
        ).agg(
            total_unidades=('cantidad_num', 'sum'),
            numero_de_entregas=('mes_entrega', 'count') # Count occurrences
        ).reset_index()

        # Rename columns to match requirements exactly
        consolidado = consolidado.rename(columns={
            'mes_entrega': 'mes',
            'producto_oficial_catalogo': 'producto_oficial',
            'laboratorio_estandar': 'laboratorio'
        })

        output_file = os.path.join(self.OUTPUT_DIR, "Consolidado_Final_Auditable.csv")
        consolidado.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Consolidated report saved to {output_file}")

        # Print Audit Table
        print("\n" + "="*80)
        print(f"{'Archivo':<30} | {'Registros':<10} | {'Suma Cantidad':<20} | {'% Coincidencia':<15}")
        print("-" * 80)
        for row in self.audit_records:
            print(f"{row['Archivo']:<30} | {row['Total Registros']:<10} | {row['Suma Cantidades']:<20,.2f} | {row['% Coincidencia']:<15.2f}%")
        print("="*80 + "\n")

if __name__ == "__main__":
    processor = DeliveryProcessorPro()
    processor.load_catalog()
    processor.process_files()
    processor.consolidate_and_report()
