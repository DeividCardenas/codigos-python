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
        B: Keyword search (LAB_MAP)
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

        # Rule B: Keywords from LAB_MAP
        text_upper = text.upper()
        for key, val in self.LAB_MAP.items():
            # Check for the key (alias) in text
            # Use strict word boundary if key is short to avoid false positives?
            # User example keys are distinct enough (TQ, MK might need care).
            # We will search for the key string.
            if key in text_upper:
                return val

        # Rule C
        return "NO_IDENTIFICADO"

    def normalize_key(self, row):
        """
        Create the cross-reference key (llave_de_cruce).
        Accepts the whole row to check quantity as well.
        """
        text = row.get('descripcion_original')
        qty = row.get('cantidad')

        # Check for missing description or quantity
        # Quantity check: Assuming 0 is "missing" or literal 0 deliveries which are invalid for processing usually.
        # But we must preserve the row.
        if pd.isna(qty) or qty == 0:
            return "POR_REVISAR"

        if pd.isna(text) or text == "":
            return "POR_REVISAR"

        text = str(text)

        # 1. Basic Clean
        clean = self.remove_accents(text).upper()

        # 2. Remove pipes and other noise, keep alphanumeric, space, %, .
        clean = re.sub(r'[^A-Z0-9\s\.\%]', ' ', clean)

        # 3. Normalize Units (Remove spaces: 500 MG -> 500MG)
        # Regex to find Number + Space + Unit
        # Added flags=re.IGNORECASE just in case, though we uppercased already.
        unit_pattern = r'(\d)\s+(' + '|'.join(self.KNOWN_UNITS) + r')\b'
        clean = re.sub(unit_pattern, r'\1\2', clean, flags=re.IGNORECASE)

        # 4. Collapse multiple spaces
        clean = re.sub(r'\s+', ' ', clean).strip()

        # 5. Check "POR_REVISAR" criteria (Structure check)
        if not clean:
            return "POR_REVISAR"

        # We must allow digits before the unit because we just removed the space (e.g., '500MG')
        search_pattern = r'(?:^|[\s\d])(' + '|'.join(self.KNOWN_UNITS) + r')\b'
        if not re.search(search_pattern, clean, flags=re.IGNORECASE):
            return "POR_REVISAR"

        return clean

    def clean_quantity_colombian(self, val):
        """
        Parses a string quantity assuming strict Colombian formatting:
        "1.000,50" -> 1000.50
        """
        if not isinstance(val, str):
            return val
        val = val.replace('.', '')
        val = val.replace(',', '.')
        return val

    def load_files(self, pattern="*.csv"):
        """
        Load all CSV files matching pattern.
        """
        files = glob.glob(pattern)
        logging.info(f"Found {len(files)} files matching '{pattern}'")

        # Filter out known report/output files
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
                # Try reading with different separators
                try:
                    df = pd.read_csv(f, dtype=str, on_bad_lines='skip')
                except:
                    df = pd.read_csv(f, sep=';', dtype=str, on_bad_lines='skip')

                col_map = self.detect_columns(df)
                if not col_map:
                    logging.warning(f"Skipping {f}: Missing required columns.")
                    continue

                # Rename columns
                df = df.rename(columns={
                    col_map['descripcion_original']: 'descripcion_original',
                    col_map['cantidad']: 'cantidad'
                })

                # Add Metadata (mes_entrega)
                # Extract simple filename without extension
                fname_clean = os.path.splitext(os.path.basename(f))[0]
                df['mes_entrega'] = fname_clean

                # Clean quantity
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
        self.processed_df = self.raw_df.copy()

        # Extract Lab
        self.processed_df['Laboratorio'] = self.processed_df['descripcion_original'].apply(self.extract_lab)

        # Create Key (Pass row)
        self.processed_df['llave_de_cruce'] = self.processed_df.apply(self.normalize_key, axis=1)

        # 3. Save Individual Processed Files
        output_dir = "procesados"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        months = self.processed_df['mes_entrega'].unique()
        audit_data = []

        for m in months:
            m_df = self.processed_df[self.processed_df['mes_entrega'] == m].copy()
            m_total_qty = m_df['cantidad'].sum()
            m_count = len(m_df)

            # Save individual file
            out_name = f"{m}_LIMPIO.csv"
            out_path = os.path.join(output_dir, out_name)
            m_df.to_csv(out_path, index=False, encoding='utf-8-sig')

            audit_data.append({
                'Nombre del Archivo': m,
                'Registros Procesados': m_count,
                'Suma Total Cantidad': m_total_qty
            })

        # 4. Consolidate Master
        logging.info("Consolidating Master...")

        # Group by Key + Lab
        self.consolidated_df = self.processed_df.groupby(['llave_de_cruce', 'Laboratorio']).agg(
            Cantidad_Total_Entregada=('cantidad', 'sum'),
            Numero_de_Entregas_Asociadas=('cantidad', 'count')
        ).reset_index()

        self.consolidated_df = self.consolidated_df.rename(columns={'llave_de_cruce': 'Medicamento_Estandarizado'})

        # 5. Integrity Validation
        total_qty_final = self.consolidated_df['Cantidad_Total_Entregada'].sum()

        # Using a small epsilon
        assert abs(total_qty_initial - total_qty_final) < 0.01, \
            f"Integrity Check Failed! Initial: {total_qty_initial}, Final: {total_qty_final}"

        # 6. Export Master
        master_file = "Consolidado_Total_Entregas.csv"
        self.consolidated_df.to_csv(master_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved master consolidated report to {master_file}")

        # 7. Print Audit Summary
        print("\n" + "="*60)
        print("RESUMEN DE CONTROL DE INTEGRIDAD")
        print("="*60)
        print(f"{'Nombre del Archivo':<30} | {'Registros':<10} | {'Suma Cantidad':<20}")
        print("-" * 65)
        for row in audit_data:
            print(f"{row['Nombre del Archivo']:<30} | {row['Registros Procesados']:<10} | {row['Suma Total Cantidad']:<20,.2f}")
        print("-" * 65)
        print(f"{'TOTAL GENERAL':<30} | {len(self.processed_df):<10} | {total_qty_final:<20,.2f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    sanitizer = DeliverySanitizer()
    sanitizer.load_files()
    sanitizer.run_pipeline()
