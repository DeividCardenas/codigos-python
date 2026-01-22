# Documentación del Sistema de Procesamiento y Reportes (GenHospi)

Este documento sirve como la guía oficial para la operación de las herramientas de limpieza de datos y generación de reportes ejecutivos: `delivery_processor_pro.py` y `report_generator.py`.

## 1. Resumen del Ecosistema de Datos

El sistema está diseñado para transformar datos crudos de dispensación de medicamentos en información estratégica para la toma de decisiones.

### Archivos de Entrada
*   **Archivos de Eventos:** Archivos CSV mensuales que contienen los registros de dispensación. Deben seguir la nomenclatura cronológica para asegurar el ordenamiento correcto (ej. `6junioEvento.csv`, `7julioEvento.csv`, ..., `11noviembreEvento.csv`).
*   **Catálogo Maestro:** El archivo `productos - Hoja 1.csv` que actúa como la fuente de verdad para normalizar nombres de medicamentos y laboratorios.

### Flujo de Datos
1.  **Entrada Cruda:** Recepción de archivos CSV con descripciones heterogéneas.
2.  **Limpieza y Validación (`delivery_processor_pro.py`):** Normalización de textos, estandarización de laboratorios y cruce difuso (fuzzy matching) contra el catálogo maestro.
3.  **Consolidado:** Generación de una base de datos maestra unificada (`Consolidado_Final_Auditable.csv`).
4.  **Reportes Visuales (`report_generator.py`):** Creación de rankings, gráficas y KPIs de gestión.

## 2. Requisitos del Sistema

El sistema requiere **Python 3.8+** y las siguientes librerías especializadas para análisis de datos y visualización.

### Instalación de Dependencias
Ejecute el siguiente comando en su terminal:

```bash
pip install pandas matplotlib seaborn openpyxl
```

## 3. Guía de Ejecución

Para procesar la información desde cero, ejecute los scripts en el siguiente orden estricto:

### Paso 1: Limpieza y Consolidación
Procesa los archivos mensuales, limpia los datos y genera el archivo maestro.

```bash
python delivery_processor_pro.py
```

### Paso 2: Generación de Reportes
Toma el consolidado generado en el paso anterior y crea las visualizaciones y el Excel de ranking.

```bash
python report_generator.py
```

## 4. Estructura de Salida

Al finalizar la ejecución, el sistema generará los siguientes recursos:

*   **Carpeta `/procesados/`:** Contiene los archivos CSV individuales de cada mes, ya limpios y auditables (ej. `6junioEvento_LIMPIO.csv`).
*   **Archivo `Consolidado_Final_Auditable.csv`:** La base de datos unificada que sirve como insumo para el generador de reportes.
*   **Carpeta `/reportes_visuales/`:** Contiene los entregables para la gerencia y el cliente:
    *   `Ranking_Top50_Medicamentos.xlsx` (Excel con pestañas mensual y global).
    *   `Pareto_Top30_Medicamentos.png` (Gráfico de barras horizontales).
    *   `Top10_Laboratorios.png` (Gráfico de participación de mercado).
    *   `Evolucion_Mensual_Operativa.png` (Gráfico lineal comparativo).
*   **Archivo `Indicadores_Gestion.txt`:** Resumen de texto con KPIs operativos (promedios, variación porcentual).

## 5. Notas de Mantenimiento

### Actualización del Catálogo Maestro
Si ingresan nuevos medicamentos al mercado o al contrato:
1.  Abra el archivo `productos - Hoja 1.csv`.
2.  Agregue las nuevas referencias manteniendo el formato de las columnas existentes.
3.  Guarde el archivo y vuelva a ejecutar `delivery_processor_pro.py` para que los nuevos datos se reconozcan en el cruce.

### Nomenclatura de Archivos
Es crucial mantener la estructura de nombres de los archivos mensuales (ej. iniciar con el número del mes `6...`, `7...`) para que el script `report_generator.py` pueda ordenar cronológicamente la evolución mensual y calcular correctamente las variaciones de crecimiento o decrecimiento.
