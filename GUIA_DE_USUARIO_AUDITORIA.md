# Auditoría de Medicamentos - Guía de Usuario

## 1. Introducción y Objetivo

Este sistema está diseñado para realizar una **auditoria determinística** del cruce de medicamentos entre los reportes de entregas mensuales y el Contrato Maestro.

A diferencia de los sistemas tradicionales que usan "probabilidades" (cruce difuso o estadístico), este motor utiliza reglas estrictas de química y farmacología para garantizar la precisión. El objetivo es identificar con certeza si lo que se está entregando cumple al 100% con lo pactado, o si existen desviaciones en la presentación o la composición.

## 2. Diccionario de Interpretación (Estado_Match)

El reporte de auditoría (`auditoria_matches_YYYYMMDD.csv`) genera una columna clave llamada **Estado_Match**. Utilice la siguiente tabla para interpretar los resultados y tomar acción:

| Estado | Significado | Acción Recomendada |
| :--- | :--- | :--- |
| **MATCH EXACTO** | El medicamento entregado coincide al 100% con el contrato en **Componentes Químicos**, **Dosis Exacta** y **Forma Farmacéutica**. | ✅ **Ninguna**. El cruce es perfecto y no requiere revisión. |
| **CAMBIO DE PRESENTACIÓN** | Los componentes químicos y la dosis son idénticos, pero el **empaque** es diferente (Ej. El contrato pide *Tabletas* y se entregó en *Cápsulas*). | ⚠️ **Verificar**. Revise si el cambio de presentación es aceptable por costos o logística. |
| **SUGERENCIA DE SUSTITUCIÓN** | La **Dosis** es exacta y el **Componente Base** coincide, pero hay diferencias en los componentes secundarios (Ej. *Acetaminofén+Hidrocodona* vs *Acetaminofén+Oxicodona*). | ⚕️ **Consultar Médica**. Valide con el área clínica si la sustitución terapéutica es viable. |
| **NO ENCONTRADO** | No se encontró ninguna coincidencia viable. La dosis no cuadra o la molécula es totalmente distinta. | ❌ **Investigar**. Puede ser un producto nuevo fuera de contrato o un error de digitación en el archivo de entrada. |

## 3. Funcionamiento Interno (Normalización)

El sistema utiliza un módulo de "Lógica Estricta" (`StrictMatcher`) que procesa los textos antes de compararlos.

### A. Regla Química (A+B = B+A)
El sistema entiende que el orden de los factores no altera el producto.
*   **Entrada:** `Codeina + Acetaminofen`
*   **Contrato:** `Acetaminofen + Codeina`
*   **Resultado:** El sistema ordena alfabéticamente ambos (`ACETAMINOFEN + CODEINA`) y detecta que son **QUÍMICAMENTE IDÉNTICOS**.

### B. Protección de Presentación (Empaque)
A diferencia de sistemas anteriores que eliminaban palabras como "TAB" para limpiar el texto, este motor **protege y normaliza** la forma farmacéutica, ya que es crítica para el contrato.

Se utiliza un diccionario técnico (`logic/medication_parser.py`) para estandarizar variaciones. Ejemplos:

| Categoría | Variaciones en Entrada (Input) | Término Normalizado (Sistema) |
| :--- | :--- | :--- |
| **TABLETAS** | `TAB`, `TAB REC`, `TAB LIB PROL`, `GRAGEA`, `COMP` | **TABLETA** |
| **CÁPSULAS** | `CAP`, `CAP BLAND`, `CAPSULA BLANDA`, `CAP DURA` | **CAPSULA** |
| **INYECTABLES** | `AMP`, `AMPOLLA`, `VIAL`, `JERINGA`, `PRELLENADA` | **INYECTABLE** |
| **LÍQUIDOS** | `JARABE`, `JBE`, `SUSPENSION`, `GOTAS`, `ELIXIR` | **SOLUCION** |
| **UNIDADES** | `FRASCO`, `SOBRE`, `BOLSA`, `KIT`, `CAJA` | **UNIDAD** |

*Si el contrato pide `TABLETA` y llega `TAB REC`, el sistema lo marca como **MATCH EXACTO** (porque son la misma categoría). Si llega `CAPSULA`, lo marca como **CAMBIO DE PRESENTACIÓN**.*

## 4. Flujo de Trabajo Recomendado

1.  **Preparación:**
    *   Asegúrese de tener el archivo `Targets.csv` (Contrato) en la carpeta raíz.
    *   Coloque los archivos de movimiento mensual en la carpeta raíz (ej. `10octubreEvento.csv`).

2.  **Ejecución:**
    *   Corra el script principal: `python cruce_entregas_capita.py`

3.  **Análisis del Reporte:**
    *   Abra el archivo generado: `auditoria_matches_YYYYMMDD.csv`.
    *   Active los **Filtros** en Excel.
    *   **Paso 1:** Filtre `Estado_Match` y desmarque "MATCH EXACTO". Concéntrese en lo demás.
    *   **Paso 2:** Revise los "CAMBIO DE PRESENTACIÓN". ¿Son aceptables?
    *   **Paso 3:** Revise las "SUGERENCIA DE SUSTITUCIÓN" en la columna `Observaciones_Auditoria`.

## 5. FAQ (Preguntas Frecuentes)

**P: ¿Por qué "Acetaminofén 500mg" no cruzó con "Acetaminofén 600mg"?**
R: Porque la validación de dosis es estricta. Si los números no coinciden exactamente, se marca como **NO ENCONTRADO** para evitar errores de medicación.

**P: ¿El sistema detecta si escribí "Acetaminofén" con tilde o sin tilde?**
R: Sí. El sistema elimina acentos y normaliza mayúsculas automáticamente antes de comparar.

**P: ¿Por qué el "Adol Pro" me sale como Sugerencia y no como Match?**
R: Probablemente porque el contrato tiene una sal específica (ej. Oxicodona) y el producto de entrada tiene otra (ej. Hidrocodona). Aunque la base (Acetaminofén) y la dosis sean iguales, químicamente son distintos, por lo que se alerta como una **SUGERENCIA DE SUSTITUCIÓN**.
