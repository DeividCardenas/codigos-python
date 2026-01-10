# Sistema de Cruce de Entregas Farmacéuticas (Advanced Matching)

Este sistema permite cruzar archivos de entregas de medicamentos (archivos "Evento") contra un maestro de productos ("Targets"), utilizando algoritmos avanzados para maximizar el porcentaje de coincidencia incluso con errores de ortografía, abreviaturas o diferencias en la descripción.

## Características Principales

*   **Coincidencia Híbrida Inteligente:**
    1.  **Código Exacto:** Prioridad máxima si el `codigo_neg` o `CUM` coinciden.
    2.  **Búsqueda Semántica (TF-IDF):** Utiliza vectores de texto y N-gramas para encontrar similitudes incluso si las palabras están desordenadas o mal escritas (ej. "Acetaminofen 500mg Tab" vs "Tab. 500 mg Acetaminofén").
    3.  **Validación de Dosis:** Un algoritmo de seguridad verifica que las dosis (números) coincidan para evitar falsos positivos (ej. evita cruzar "Losartan 50 mg" con "Losartan 100 mg").
*   **Alto Rendimiento:** Procesamiento por lotes (batch processing) que permite manejar millones de registros en segundos/minutos.
*   **Normalización Automática:** Limpieza de texto, acentos, y estandarización de unidades (mg, ml, g) y abreviaturas farmacéuticas (TAB -> TABLETA, AMP -> AMPOLLA).

## Requisitos

El sistema requiere Python 3.8+ y las siguientes librerías:

```bash
pip install pandas numpy rapidfuzz scikit-learn tqdm psutil
```

## Estructura de Archivos

*   **Targets (Maestro):** Debe ser un CSV (`Targets.csv`) con al menos una columna de `codigo_neg` y opcionalmente `descripcion`.
*   **Archivos de Entrada (Eventos):** Archivos CSV con los movimientos. El sistema detecta automáticamente columnas como `codigo_neg`, `descripcion`, `cantidad_entregada`.

## Uso

Para ejecutar el cruce automáticamente con todos los archivos CSV en la carpeta que contengan "Evento" en el nombre:

```bash
python3 cruce_entregas_capita.py
```

Para especificar archivos puntuales:

```bash
python3 cruce_entregas_capita.py --files AbrilEvento.csv MayoEvento.csv
```

Para especificar un archivo de targets diferente:

```bash
python3 cruce_entregas_capita.py --targets MiMaestro.csv
```

### Opciones Avanzadas

*   `--batch_size`: Ajusta el tamaño del lote para procesamiento en memoria (defecto: 5000). Útil para optimizar RAM.

## Reportes Generados

1.  **`reporte_cruce_avanzado.csv`**: Tabla dinámica con los productos del maestro (filas) y las cantidades entregadas por mes/archivo (columnas). Incluye una columna de total acumulado.
2.  **`no_encontrados_avanzado_AAAAMMDD.csv`**: Listado detallado de los registros que no lograron cruzarse con el maestro, útil para depuración y actualización del maestro.
3.  **Resumen en Consola**: Muestra el porcentaje de cobertura total y estadísticas por archivo.

## Estrategia de Optimización

El algoritmo sigue este flujo para cada registro de entrada:

1.  **Limpieza:** Estandariza el texto (quita "LABORATORIO X", expande "TAB", quita acentos).
2.  **Indexación:** Si existe código, busca en un índice hash (O(1)).
3.  **Vectorización:** Si no hay código, convierte la descripción en un vector numérico TF-IDF.
4.  **Búsqueda:** Encuentra los 5 vecinos más cercanos en el espacio vectorial.
5.  **Re-ranking:** Refina los candidatos usando distancia de edición (Levenshtein) y validación de características numéricas.
6.  **Decisión:** Acepta el match si el puntaje supera el umbral de confianza (88%).

---
**Nota:** Si los archivos de entrada solo contienen códigos (sin descripciones), el sistema funcionará en modo "solo código" de máxima velocidad. Para aprovechar la búsqueda difusa, asegúrese de incluir columnas de descripción en los archivos de entrada.
