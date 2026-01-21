import pandas as pd
import numpy as np
import glob
import re
import unicodedata
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeliverySanitizer:
    LAB_KEYWORDS = [
        'GENFAR', 'LAPROFF', 'ECAR', 'TECNOQUIMICAS', 'LA SANTE',
        'SIEGFRIED', 'NOVAMED', 'BAYER', 'PROCAPS'
    ]

    # Strictly defined known units per user requirement
    KNOWN_UNITS = ['MG', 'ML', 'MCG', 'TABLETA', 'CAPSULA']

    def __init__(self):
        self.raw_df = pd.DataFrame()
        self.processed_df = pd.DataFrame()
        self.consolidated_df = pd.DataFrame()

    def remove_accents(self, text):
        if not isinstance(text, str):
            return str(text)
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def detect_columns(self, df):
        """
        Identify 'nombre' and 'cantidad' columns allowing for variations.
        Returns a map or None if columns are missing.
        """
        cols = [c.lower() for c in df.columns]

        col_map = {}

        # Detect Description
        desc_candidates = ['nombre', 'nombres', 'descripcion', 'producto']
        for cand in desc_candidates:
            matches = [c for c in df.columns if cand in c.lower()]
            if matches:
                col_map['descripcion_original'] = matches[0]
                break

        # Detect Quantity
        qty_candidates = ['cantidad', 'cantidades', 'unidades', 'cant']
        for cand in qty_candidates:
            matches = [c for c in df.columns if cand in c.lower()]
            if matches:
                col_map['cantidad'] = matches[0]
                break

        if 'descripcion_original' in col_map and 'cantidad' in col_map:
            return col_map
        return None

    def extract_lab(self, text):
        """
        Extract Laboratory based on rules:
        A: Between pipes | LAB |
        B: Keyword search
        C: NO_IDENTIFICADO
        """
        if not isinstance(text, str):
            return "NO_IDENTIFICADO"

        # Rule A: Pipes
        if '|' in text:
            parts = text.split('|')
            if len(parts) >= 2:
                potential_lab = parts[1].strip()
                if potential_lab:
                    return potential_lab.upper()

        # Rule B: Keywords
        text_upper = text.upper()
        for keyword in self.LAB_KEYWORDS:
            if keyword in text_upper:
                return keyword

        # Rule C
        return "NO_IDENTIFICADO"

    def normalize_key(self, text):
        """
        Create the cross-reference key (llave_de_cruce).
        - Uppercase
        - Remove accents
        - Remove special chars (keep letters, numbers, %, dots)
        - Normalize units (remove space between number and unit: 500 MG -> 500MG)
        """
        if pd.isna(text) or text == "":
            return "POR_REVISAR"

        # 1. Basic Clean
        clean = self.remove_accents(str(text)).upper()

        # 2. Remove pipes and other noise, keep alphanumeric, space, %, .
        # Note: We remove special chars like *, |, etc.
        clean = re.sub(r'[^A-Z0-9\s\.\%]', ' ', clean)

        # 3. Normalize Units (Remove spaces: 500 MG -> 500MG)
        # Regex to find Number + Space + Unit
        unit_pattern = r'(\d)\s+(' + '|'.join(self.KNOWN_UNITS) + r')\b'
        clean = re.sub(unit_pattern, r'\1\2', clean)

        # 4. Collapse multiple spaces
        clean = re.sub(r'\s+', ' ', clean).strip()

        # 5. Check "POR_REVISAR" criteria
        # - Empty
        if not clean:
            return "POR_REVISAR"

        # - No known unit present? (Strict check: Regex with word boundaries)
        # We must allow digits before the unit because we just removed the space (e.g., '500MG')
        # \b fails between '0' and 'M' because both are word characters.
        search_pattern = r'(?:^|[\s\d])(' + '|'.join(self.KNOWN_UNITS) + r')\b'
        if not re.search(search_pattern, clean):
            return "POR_REVISAR"

        return clean

    def clean_quantity_colombian(self, val):
        """
        Parses a string quantity assuming strict Colombian formatting:
        - '.' is a thousands separator (Removed)
        - ',' is a decimal separator (Replaced by '.')

        Example: "1.000,50" -> 1000.50

        If the value is not a string (e.g. already float/int), it is returned as is.
        """
        if not isinstance(val, str):
            return val

        # Remove dots (thousands)
        val = val.replace('.', '')
        # Replace commas with dots (decimals)
        val = val.replace(',', '.')

        return val

    def load_files(self, pattern="*.csv"):
        """
        Load all CSV files matching pattern, skip those without required columns.
        """
        files = glob.glob(pattern)
        logging.info(f"Found {len(files)} files matching '{pattern}'")

        # Filter out known report/output files to avoid double counting
        excluded_prefixes = ['no_encontrados', 'top_missing', 'reporte', 'REPORTE', 'Consolidado', 'Targets']
        filtered_files = []
        for f in files:
            fname = os.path.basename(f)
            if any(fname.startswith(p) for p in excluded_prefixes):
                logging.info(f"Skipping report/target file: {f}")
                continue
            filtered_files.append(f)

        dfs = []

        for f in filtered_files:
            try:
                # Try reading with different separators if needed, but assuming comma/pipe handling
                # First try standard read (comma default)
                try:
                    df = pd.read_csv(f, dtype=str, on_bad_lines='skip')
                except:
                    # Fallback to semicolon
                    df = pd.read_csv(f, sep=';', dtype=str, on_bad_lines='skip')

                col_map = self.detect_columns(df)
                if not col_map:
                    logging.warning(f"Skipping {f}: Missing required columns (nombre/cantidad).")
                    continue

                # Rename columns for consistency
                df = df.rename(columns={
                    col_map['descripcion_original']: 'descripcion_original',
                    col_map['cantidad']: 'cantidad'
                })

                # Keep only relevant columns plus others if needed?
                # Request says "Identificar ... y renombrarlas ... preservar estructura original"
                # We will keep all, but ensure our key columns exist.

                # Add Metadata
                df['mes_archivo'] = os.path.splitext(os.path.basename(f))[0]

                # Clean quantity using explicit helper
                df['cantidad'] = df['cantidad'].apply(self.clean_quantity_colombian)
                df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce').fillna(0)

                dfs.append(df)
                logging.info(f"Loaded {f}: {len(df)} rows")

            except Exception as e:
                logging.error(f"Error reading {f}: {e}")

        if dfs:
            self.raw_df = pd.concat(dfs, ignore_index=True)
            logging.info(f"Total raw rows loaded: {len(self.raw_df)}")
        else:
            logging.warning("No valid dataframes loaded.")

    def run_pipeline(self):
        if self.raw_df.empty:
            logging.error("No data to process.")
            return

        # 1. Validation Pre-Clean
        total_qty_initial = self.raw_df['cantidad'].sum()
        logging.info(f"Total Initial Quantity: {total_qty_initial:,.2f}")

        # 2. Sanitize / Normalize
        logging.info("Starting normalization...")

        # Use assign to create new columns without mutating original messy structure too much
        # However, we concatenated everything into raw_df. We will work on a copy or directly.

        self.processed_df = self.raw_df.copy()

        # Extract Lab
        self.processed_df['Laboratorio'] = self.processed_df['descripcion_original'].apply(self.extract_lab)

        # Create Key
        self.processed_df['llave_de_cruce'] = self.processed_df['descripcion_original'].apply(self.normalize_key)

        # 3. Consolidation
        logging.info("Consolidating...")

        # Group by Key + Lab
        # Aggregation: Sum Quantity, Count Occurrences

        self.consolidated_df = self.processed_df.groupby(['llave_de_cruce', 'Laboratorio']).agg(
            Cantidad_Total_Entregada=('cantidad', 'sum'),
            Numero_de_Entregas_Asociadas=('cantidad', 'count') # Count rows
        ).reset_index()

        # Rename for final output requirements: Medicamento_Estandarizado
        self.consolidated_df = self.consolidated_df.rename(columns={'llave_de_cruce': 'Medicamento_Estandarizado'})

        # 4. Integrity Validation
        total_qty_final = self.consolidated_df['Cantidad_Total_Entregada'].sum()
        logging.info(f"Total Final Quantity: {total_qty_final:,.2f}")

        # Using a small epsilon for float comparison just in case, though pandas sum should be consistent
        assert abs(total_qty_initial - total_qty_final) < 0.01, \
            f"Integrity Check Failed! Initial: {total_qty_initial}, Final: {total_qty_final}"

        logging.info("Integrity Check PASSED.")

        # 5. Export
        output_file = "Consolidado_Entregas_Limpio.csv"
        self.consolidated_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved consolidated report to {output_file}")


if __name__ == "__main__":
    sanitizer = DeliverySanitizer()
    sanitizer.load_files()
    sanitizer.run_pipeline()
