import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import numpy as np

# Configure plotting style
sns.set_theme(style="whitegrid")

class ReportGenerator:
    def __init__(self, input_file="Consolidado_Final_Auditable.csv"):
        self.input_file = input_file
        self.output_dir = "reportes_visuales"
        self.df = None

        # Month mapping: Number -> Name
        self.month_map = {
            1: "ENERO", 2: "FEBRERO", 3: "MARZO", 4: "ABRIL",
            5: "MAYO", 6: "JUNIO", 7: "JULIO", 8: "AGOSTO",
            9: "SEPTIEMBRE", 10: "OCTUBRE", 11: "NOVIEMBRE", 12: "DICIEMBRE"
        }

    def setup_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

    def parse_month(self, month_str):
        """
        Extracts month number from string like '6junioEvento' or '10octubreEvento'.
        Returns (order, name).
        """
        # Extract first number found
        match = re.search(r'^(\d+)', str(month_str))
        if match:
            month_num = int(match.group(1))
            month_name = self.month_map.get(month_num, f"MES_{month_num}")
            return month_num, month_name
        return 99, str(month_str)

    def load_data(self):
        if not os.path.exists(self.input_file):
            print(f"Error: Input file {self.input_file} not found.")
            return False

        try:
            print(f"Loading data from {self.input_file}...")
            self.df = pd.read_csv(self.input_file)

            # Enrich data with month order and standard name
            # Apply parsing
            parsed = self.df['mes'].apply(self.parse_month)
            self.df['mes_orden'] = parsed.apply(lambda x: x[0])
            self.df['mes_nombre'] = parsed.apply(lambda x: x[1])

            # Sort chronologically
            self.df = self.df.sort_values(by='mes_orden')

            print(f"Data loaded: {len(self.df)} rows.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def generate_pareto_chart(self):
        print("Generating Pareto chart...")
        # Aggregate Global
        global_group = self.df.groupby('producto_oficial')['total_unidades'].sum().reset_index()
        top30 = global_group.sort_values(by='total_unidades', ascending=False).head(30)

        # Calculate Cumulative %
        total_volume = global_group['total_unidades'].sum()
        top30['cumulative_sum'] = top30['total_unidades'].cumsum()
        top30['cumulative_pct'] = (top30['cumulative_sum'] / total_volume) * 100

        plt.figure(figsize=(14, 10))

        # Bar Plot
        ax1 = sns.barplot(x='total_unidades', y='producto_oficial', data=top30, palette='viridis', hue='producto_oficial', legend=False)
        ax1.set_xlabel('Total Unidades')
        ax1.set_ylabel('Producto')
        ax1.set_title('Top 30 Productos - Análisis de Pareto (Volumen)', fontsize=16)

        # Add values to bars
        for i, v in enumerate(top30['total_unidades']):
            ax1.text(v + (v * 0.01), i, f'{int(v):,}', va='center', fontsize=9)

        # Cumulative Line
        ax2 = ax1.twiny()
        # We need to match the y-axis. The bar plot has 0..29.
        # Line plot needs to align.
        # It's tricky to mix horizontal bars with a line plot on the same categorical axis in basic matplotlib/seaborn
        # without some hacking.
        # Standard Pareto is Vertical Bars + Line.
        # Requirement: "Pareto (Barras Horizontales): Visualiza los 30 medicamentos más críticos... nombres sean legibles."
        # Requirement: "Incluye una línea de porcentaje acumulado".
        # Vertical names are hard to read. Horizontal is better.
        # Adding a line to horizontal bars: The "Line" becomes a curve along the Y axis.

        # Let's plot the cumulative points as a line on the secondary x-axis (top),
        # mapping to the same y-coordinates (0 to 29).
        ax2.plot(top30['cumulative_pct'], range(len(top30)), color='red', marker='o', linestyle='-', linewidth=2, label='Acumulado %')
        ax2.set_xlim(0, 100) # Percentage is 0-100 (or close to it, though strictly it's share of global)
        # Note: If top 30 is only 50% of global, the line won't reach 100. It refers to cumulative of the dataset shown?
        # Usually Pareto is cumulative of the WHOLE, but we are only showing Top 30.
        # User said: "qué productos representan el 80% del movimiento". So it implies cumulative of the TOTAL volume.

        ax2.set_xlabel('Porcentaje Acumulado del Total Global (%)', color='red')
        ax2.tick_params(axis='x', labelcolor='red')

        # Add percentage labels on the line points
        for i, v in enumerate(top30['cumulative_pct']):
            ax2.text(v + 1, i, f'{v:.1f}%', color='red', va='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Pareto_Top30_Medicamentos.png'), dpi=300)
        plt.close()

    def generate_lab_chart(self):
        print("Generating Lab concentration chart...")
        lab_group = self.df.groupby('laboratorio')['total_unidades'].sum().reset_index()
        top10_labs = lab_group.sort_values(by='total_unidades', ascending=False).head(10)

        plt.figure(figsize=(10, 8))

        # Pie Chart as requested "Torta o Barras". Pie is good for "Participación".
        # Let's use a Donut chart for a modern look.
        colors = sns.color_palette('pastel')[0:10]

        # We need to handle "Others" to make a real 100% pie?
        # User said: "participación de los 10 laboratorios principales en el volumen total."
        # If we only plot top 10, the pie is incomplete.
        # Better to add "OTROS".
        total_vol = lab_group['total_unidades'].sum()
        top10_vol = top10_labs['total_unidades'].sum()
        others_vol = total_vol - top10_vol

        # Create data for pie
        pie_data = top10_labs.copy()
        if others_vol > 0:
            new_row = pd.DataFrame({'laboratorio': ['OTROS'], 'total_unidades': [others_vol]})
            pie_data = pd.concat([pie_data, new_row], ignore_index=True)

        plt.pie(pie_data['total_unidades'], labels=pie_data['laboratorio'], colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85)

        # Draw circle for Donut
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        plt.title('Participación de Mercado - Top 10 Laboratorios', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Top10_Laboratorios.png'), dpi=300)
        plt.close()

    def generate_evolution_chart(self):
        print("Generating Monthly Evolution chart...")
        monthly_stats = self.df.groupby(['mes_orden', 'mes_nombre']).agg(
            total_unidades=('total_unidades', 'sum'),
            numero_de_entregas=('numero_de_entregas', 'sum')
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # X Axis
        x = range(len(monthly_stats))
        labels = monthly_stats['mes_nombre']

        # Axis 1: Units (Bars or Line? User said "gráfico de líneas comparativo que muestre dos ejes")
        # Line 1
        color = 'tab:blue'
        ax1.set_xlabel('Mes')
        ax1.set_ylabel('Total Unidades', color=color)
        ax1.plot(x, monthly_stats['total_unidades'], color=color, marker='o', linewidth=3, label='Unidades')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)

        # Add values
        for i, v in enumerate(monthly_stats['total_unidades']):
            ax1.text(i, v, f'{int(v):,}', color=color, ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Axis 2: Deliveries
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Número de Entregas (Transacciones)', color=color)
        ax2.plot(x, monthly_stats['numero_de_entregas'], color=color, marker='s', linestyle='--', linewidth=3, label='Entregas')
        ax2.tick_params(axis='y', labelcolor=color)

        # Add values
        for i, v in enumerate(monthly_stats['numero_de_entregas']):
            ax2.text(i, v, f'{int(v):,}', color=color, ha='center', va='top', fontsize=9, fontweight='bold')

        plt.title('Evolución Mensual: Unidades vs Transacciones', fontsize=16)
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Evolucion_Mensual_Operativa.png'), dpi=300)
        plt.close()

    def generate_kpi_report(self):
        print("Generating KPI report...")
        output_file = "Indicadores_Gestion.txt"

        # 1. Average units per delivery
        total_units = self.df['total_unidades'].sum()
        total_deliveries = self.df['numero_de_entregas'].sum()
        avg_units_per_delivery = total_units / total_deliveries if total_deliveries > 0 else 0

        # 2. Percentage Variation
        # Get first and last month
        monthly_stats = self.df.groupby(['mes_orden', 'mes_nombre'])['total_unidades'].sum().reset_index()

        if len(monthly_stats) >= 2:
            first_month = monthly_stats.iloc[0]
            last_month = monthly_stats.iloc[-1]

            v_initial = first_month['total_unidades']
            v_final = last_month['total_unidades']

            variation_pct = ((v_final - v_initial) / v_initial) * 100 if v_initial > 0 else 0

            trend = "Crecimiento" if variation_pct >= 0 else "Contracción del volumen"
            trend_text = f"{trend} ({variation_pct:+.2f}%)"
            period_text = f"Comparativo: {first_month['mes_nombre']} vs {last_month['mes_nombre']}"
        else:
            variation_pct = 0
            trend_text = "N/A (Datos insuficientes para variación)"
            period_text = "N/A"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("INDICADORES DE GESTIÓN - GENHOSPI\n")
            f.write("===================================\n\n")

            f.write(f"1. EFICIENCIA OPERATIVA\n")
            f.write(f"   - Total Unidades Entregadas: {int(total_units):,}\n")
            f.write(f"   - Total Entregas (Transacciones): {int(total_deliveries):,}\n")
            f.write(f"   - Promedio de Unidades por Entrega: {avg_units_per_delivery:.2f}\n\n")

            f.write(f"2. EVOLUCIÓN DEL VOLUMEN\n")
            f.write(f"   - {period_text}\n")
            f.write(f"   - Variación Porcentual: {trend_text}\n")

        print(f"KPI report saved to {output_file}")

    def generate_top50_excel(self):
        output_excel = "Ranking_Top50_Medicamentos.xlsx"
        print(f"Generating Top 50 Excel report: {output_excel}...")

        try:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                # 1. Global Period
                global_group = self.df.groupby('producto_oficial').agg(
                    Total_Unidades=('total_unidades', 'sum'),
                    Frecuencia_Entregas=('numero_de_entregas', 'sum')
                ).reset_index()

                top50_global = global_group.sort_values(by='Total_Unidades', ascending=False).head(50)
                top50_global.insert(0, 'Ranking', range(1, len(top50_global) + 1))
                top50_global.to_excel(writer, sheet_name='Global_Periodo', index=False)

                # 2. Monthly Sheets
                # Get unique months in order
                months = self.df[['mes_orden', 'mes_nombre']].drop_duplicates().sort_values('mes_orden')

                for _, row in months.iterrows():
                    m_order = row['mes_orden']
                    m_name = row['mes_nombre']

                    # Filter for this month
                    month_data = self.df[self.df['mes_orden'] == m_order]

                    if month_data.empty:
                        continue

                    # Aggregate
                    month_group = month_data.groupby('producto_oficial').agg(
                        Total_Unidades=('total_unidades', 'sum'),
                        Frecuencia_Entregas=('numero_de_entregas', 'sum')
                    ).reset_index()

                    top50_month = month_group.sort_values(by='Total_Unidades', ascending=False).head(50)
                    top50_month.insert(0, 'Ranking', range(1, len(top50_month) + 1))

                    # Sheet name limited to 31 chars
                    sheet_name = m_name[:31]
                    top50_month.to_excel(writer, sheet_name=sheet_name, index=False)

            print("Excel report generated successfully.")

        except Exception as e:
            print(f"Error generating Excel report: {e}")

    def run(self):
        self.setup_directories()
        if self.load_data():
            self.generate_top50_excel()
            self.generate_pareto_chart()
            self.generate_lab_chart()
            self.generate_evolution_chart()
            self.generate_kpi_report()

if __name__ == "__main__":
    reporter = ReportGenerator()
    reporter.run()
