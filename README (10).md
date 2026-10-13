# SMIEAE · Detección de estrés y ansiedad mediante anomalías fisiológicas

Experimento de detección de anomalías sobre datos de *wearables* (Fitbit) de estudiantes universitarios. El objetivo es ver si un modelo que **solo aprende la fisiología "normal" de cada estudiante** (sus días de paz) puede señalar los días en los que el estudiante reporta **estrés o ansiedad alta** en una encuesta diaria.

El modelo base es **EOTE** (*Explainable Outlier Tree-based Ensemble*, con imputación MissForest). Se entrena con un enfoque *one-class*: solo ve bloques de 5 minutos de días tranquilos y después asigna un puntaje de anomalía a cualquier bloque nuevo.

> **Autor:** Javier Esteban Torres Sossa
> **Dataset:** SSAQS (Fitbit + encuestas diarias de estrés/ansiedad)
> **Estado:** experimental / en desarrollo

---

## Índice

1. [Contexto del experimento](#1-contexto-del-experimento)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Renombrado de archivos](#3-renombrado-de-archivos)
4. [Datos](#4-datos)
5. [Pipeline común](#5-pipeline-común)
6. [Experimentos](#6-experimentos)
7. [Resultados](#7-resultados)
8. [Dashboard](#8-dashboard)
9. [Instalación y ejecución](#9-instalación-y-ejecución)
10. [Limitaciones y pendientes](#10-limitaciones-y-pendientes)

---

## 1. Contexto del experimento

- **Pregunta:** ¿las desviaciones de la fisiología habitual de una persona (frecuencia cardiaca, HRV, SpO₂, pasos) coinciden con los días en que reporta estrés o ansiedad?
- **Unidad de análisis:** bloques de 5 minutos para el modelo, que luego se agregan a **días** para compararlos con la encuesta.
- **Etiqueta (*ground truth*):** puntaje diario de `stress` y `anxiety` (0–100) del cuestionario `daily_questions.csv`.
- **Umbrales personalizados:** cada estudiante tiene sus propios percentiles.
  - **Día de paz:** puntaje ≤ P25. Estos días se usan para entrenar.
  - **Día de crisis:** puntaje > P70. Es la clase positiva en la evaluación.
- **Supuesto de la presentación:** alrededor del 30 % del semestre corresponde a estados de estrés.

---

## 2. Estructura del repositorio

Estructura propuesta:

```
smieae-deteccion-anomalias/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_exploracion_estudiante1_eote.ipynb
│   ├── 02_estudiante1_split7030_random_forest.ipynb
│   ├── 03_poblacion_eote_estres.ipynb
│   ├── 04_poblacion_eote_carga_mental.ipynb
│   ├── 05_poblacion_eote_ansiedad.ipynb
│   └── 06_correlacion_patrones_datos_faltantes.ipynb
├── dashboard/
│   └── app_dashboard_smieae.py          # antes app5.py (lo genera el notebook 01)
├── docs/
│   └── presentacion_deteccion_anomalias_2026-06-28.pdf
├── EOTE/                                 # módulo local del modelo (Directors, Builders)
├── data/                                 # NO versionado: dataset SSAQS
│   └── SSAQS/<id_estudiante>/...
└── outputs/                              # NO versionado: CSVs generados
    ├── SMIEAE_Reporte_Completo.csv
    ├── Ranking_Completo_Anomalias.csv
    └── Ranking_Completo_Con_Caracteristicas.csv
```

---

## 3. Renombrado de archivos

| Archivo original | Nombre nuevo | Qué contiene |
|---|---|---|
| `dataset_completo.ipynb` | `notebooks/01_exploracion_estudiante1_eote.ipynb` | Exploración del estudiante 1, limpieza, EOTE sobre todo el semestre, interpretabilidad, dashboard y rankings |
| `dataset_7030.ipynb` | `notebooks/02_estudiante1_split7030_random_forest.ipynb` | Estudiante 1, split cronológico 70/30 por días, EOTE más Random Forest sobre "pilares" diarios |
| `dataset_10_estres.ipynb` | `notebooks/03_poblacion_eote_estres.ipynb` | Pipeline para 10 estudiantes; etiqueta = estrés |
| `dataset_10_estres - Copy.ipynb` | `notebooks/04_poblacion_eote_carga_mental.ipynb` | Pipeline para 10 estudiantes; entrena con estrés, etiqueta = máx(estrés, ansiedad) |
| `dataset_10_ansiedad.ipynb` | `notebooks/05_poblacion_eote_ansiedad.ipynb` | Pipeline para 10 estudiantes; etiqueta = ansiedad |
| `dataset_correlacion.ipynb` | `notebooks/06_correlacion_patrones_datos_faltantes.ipynb` | Correlación entre patrones de sensores faltantes y estrés/ansiedad (estudiantes 20–35) |
| `analisis.ipynb` (2 copias idénticas) | eliminar | Solo contiene una variable de ruta; no aporta al experimento |
| Presentación PDF | `docs/presentacion_deteccion_anomalias_2026-06-28.pdf` | Presentación "Detección de Anomalías" (28/06/2026) |
| `app5.py` | `dashboard/app_dashboard_smieae.py` | Dashboard en Streamlit |

Comandos para Windows/PowerShell. Ejecútalos en la raíz del repo después de copiar los archivos ahí:

```powershell
mkdir notebooks, dashboard, docs, outputs
git mv dataset_completo.ipynb                 notebooks/01_exploracion_estudiante1_eote.ipynb
git mv dataset_7030.ipynb                     notebooks/02_estudiante1_split7030_random_forest.ipynb
git mv dataset_10_estres.ipynb                notebooks/03_poblacion_eote_estres.ipynb
git mv "dataset_10_estres - Copy.ipynb"       notebooks/04_poblacion_eote_carga_mental.ipynb
git mv dataset_10_ansiedad.ipynb              notebooks/05_poblacion_eote_ansiedad.ipynb
git mv dataset_correlacion.ipynb              notebooks/06_correlacion_patrones_datos_faltantes.ipynb
git rm analisis.ipynb
```

> Si los archivos todavía no están en git, usa `Move-Item` en lugar de `git mv`.

---

## 4. Datos

### 4.1 Estructura esperada por estudiante

```
data/SSAQS/<id>/
├── steps.csv                     # timestamp, steps
├── oxygen.csv                    # timestamp, value (SpO2)
├── hrv.csv                       # timestamp, rmssd, low_frequency, high_frequency
├── daily_questions.csv           # timeStampStart (unix), stress, anxiety
└── Global Export Data/
    └── *heart_rate*.json         # [{dateTime: "%m/%d/%y %H:%M:%S", value: {bpm}}]
```

Todas las marcas de tiempo vienen en UTC y se convierten a hora local con un desfase fijo de **−6 h**.

### 4.2 Variables del modelo (13)

| Variable | Fuente | Cálculo (ventana de 5 min) |
|---|---|---|
| `HR_MEAN` | heart_rate JSON | media de BPM; interpolación lineal de máximo 1 bloque |
| `HR_STD` | heart_rate JSON | desviación estándar de BPM |
| `first_diff` | HR_MEAN | diferencia con el bloque anterior |
| `HR_MEAN_5` | HR_MEAN | media móvil de 5 bloques |
| `HR_STD_5` | HR_MEAN | desviación móvil de 5 bloques |
| `HRV_RMSDD`, `HRV_RMSDD_STD` | hrv.csv | media/desviación móvil (3 registros) de RMSSD; se descarta ≤0 y >200 |
| `HRV_LF`, `HRV_LF_STD` | hrv.csv | ídem para baja frecuencia; se descarta <0 |
| `HRV_HF`, `HRV_HF_STD` | hrv.csv | ídem para alta frecuencia; se descarta <0 |
| `steps` | steps.csv | suma de pasos |
| `SPO2_MEAN` | oxygen.csv | media de SpO₂; se descarta <80 |

Etiquetas: `stress` y `anxiety`, con el promedio diario de la encuesta asignado a todos los bloques de ese día.

> La versión de la presentación (sección 6.0) también usaba *activity level* (moda en 5 min) y `SPO2_STD`. Los notebooks actuales ya no los incluyen.

### 4.3 Datos faltantes (estudiante 1)

| Variable | Antes de limpiar | Después de limpiar |
|---|---|---|
| HR_MEAN | 57.93 % | 2.90 % |
| HRV_RMSSD | 90.51 % | 80.41 % |
| SPO2_MEAN | 89.24 % | 78.02 % |
| steps | 0 % | 0 % |

---

## 5. Pipeline común

1. **Carga y remuestreo** a bloques de 5 minutos de cada sensor (sección 4.2).
2. **Fusión** de todos los sensores en un `dataset_maestro` y unión con la encuesta diaria por fecha.
3. **Limpieza de "fantasmas":** se eliminan los bloques donde todas las variables fisiológicas (HR y HRV) son NaN. Así se evita que el modelo marque como anómalos bloques que solo tienen pasos.
   - Estudiante 1: 30 343 → 13 461 bloques.
4. **Filtro de adherencia:** solo se conservan los días con al menos **144 bloques**, que equivalen a unas 12 h de registro.
   - Estudiante 1: 13 461 → 12 406 bloques.
5. **Perfil psicológico personal:** se toma el máximo diario de estrés/ansiedad y se calculan P25 (paz) y P70 (crisis).
6. **Entrenamiento de EOTE** solo con bloques de días de paz (`EoteWithMissForestInTerminalBuilder`).
7. **Puntaje de anomalía** (`anomaly_score`) para cada bloque de evaluación.
8. **Agregación diaria** del puntaje (suma, picos y rachas) y **clasificación del día** como paz o crisis.

---

## 6. Experimentos

### 6.0 Presentación (versión previa) · `docs/presentacion_deteccion_anomalias_2026-06-28.pdf`

- Estudiante 1. La encuesta cubre del 2025-02-19 al 2025-06-22 (121 días con datos y 3 faltantes).
- El cruce con Fitbit deja 106 días. Después de quitar las filas que solo tenían *activity level* y pasos, quedan 13 919 bloques en 71 días útiles (rango 2025-02-21 → 2025-05-11).
- Separación en P70: 9 739 bloques normales (entrenamiento) y 4 180 de estrés. **La evaluación se hizo a nivel de bloque.**
- Se entrenó además un modelo por franja del día: mañana (3 127 bloques), tarde (4 818), noche (3 259) y madrugada (2 715).

### 6.1 `01_exploracion_estudiante1_eote.ipynb`

Exploración completa del estudiante 1:

- Diagnóstico de datos faltantes antes y después de la limpieza.
- EOTE entrenado con 3 252 bloques de paz y aplicado a **los 12 406 bloques del semestre** (sin separar entrenamiento y prueba).
- Importancia de variables por árbol de EOTE e interpretación de una instancia con `classify_and_interpret`.
- Exporta `SMIEAE_Reporte_Completo.csv`, `Ranking_Completo_Anomalias.csv` y `Ranking_Completo_Con_Caracteristicas.csv`.
- Genera y lanza el dashboard de Streamlit (sección 8).
- Lista el top 20 de días con mayor estrés reportado. El máximo es el 2025-04-21, con 66.

### 6.2 `02_estudiante1_split7030_random_forest.ipynb`

Arquitectura de dos etapas para el estudiante 1:

1. **Split cronológico por días completos:** 54 días válidos, con corte en el **2025-04-20**.
2. **Carga emocional unificada:** `MinMaxScaler` ajustado **solo con el pasado** sobre el máximo diario de estrés y ansiedad, promediados. Los umbrales también salen solo del pasado: Paz ≤ 0.2417 (P25) y Crisis > 0.5000 (P70). Los estados son Paz, Medio y Crisis.
3. **EOTE** entrenado con 2 405 bloques de paz del pasado. Asigna puntajes al pasado y al futuro.
4. **Pilares diarios.** Un bloque es anómalo si su puntaje supera el P90 del pasado (−4.90). Con eso se calculan cinco variables por día:
   - `Carga_Total_Estres`: suma de puntajes de los bloques anómalos.
   - `Severidad_Promedio`: media de puntajes de los bloques anómalos.
   - `Tren_Critico_30m`: máxima suma móvil de 6 bloques.
   - `Indice_Paz_Porcentual`: porcentaje de bloques normales.
   - `Racha_Paz_Maxima_Minutos`: racha más larga de bloques normales seguidos.
5. **Random Forest** (200 árboles, `max_depth=4`, `class_weight='balanced'`) para clasificar **Normalidad** (Paz + Medio) contra **Crisis**.

### 6.3 `03_poblacion_eote_estres.ipynb`

Pipeline con un modelo EOTE por estudiante (IDs 1–10).

- **Entrenamiento:** bloques del primer 70 % con estrés ≤ P25.
- **Etiqueta del día:** estrés máximo > P70.
- **Clasificación del día:** la suma diaria de `anomaly_score` se compara con un umbral elegido por búsqueda en rejilla (100 valores) que maximiza F1.
- Después se juntan todos los días de todos los estudiantes en una matriz global.

### 6.4 `04_poblacion_eote_carga_mental.ipynb`

Igual que 6.3 (entrena con paz de **estrés**), pero la etiqueta es la **carga mental**: `max(estrés, ansiedad)` > P70 del estrés.

### 6.5 `05_poblacion_eote_ansiedad.ipynb`

Igual que 6.3, pero tanto el entrenamiento (ansiedad ≤ P25) como la etiqueta (ansiedad > P70) usan **ansiedad**.

### 6.6 `06_correlacion_patrones_datos_faltantes.ipynb`

Revisa si **la ausencia de datos de los sensores** está relacionada con el estado emocional. Usa los estudiantes 20–35.

- Para cada bloque de 5 minutos se codifica qué sensores faltan, con el formato `BPM{0/1}_RMSSD{0/1}_SPO2{0/1}` (1 = falta).
- Se calcula el porcentaje diario de cada patrón y su correlación de Spearman con estrés y ansiedad.

---

## 7. Resultados

### 7.0 Presentación (nivel bloque, estudiante 1)

| Modelo | Exactitud | Sensibilidad | Precisión | F1 |
|---|---|---|---|---|
| Global | 63.36 % | 39.3 % | 39.3 % | 39.3 % |
| Mañana | 68.8 % | 47.9 % | 45.4 % | 46.6 % |
| Tarde | 65.6 % | 42.9 % | 44.2 % | 43.6 % |
| Noche | 76.5 % | 41.3 % | 56.0 % | 47.5 % |
| Madrugada | 66.66 % | 44.6 % | 45.9 % | 45.2 % |

### 7.1 Estudiante 1 · Random Forest sobre pilares (notebook 02)

En el entrenamiento hubo 26 días de Normalidad y 10 de Crisis. La prueba tiene 17 días inéditos:

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Normalidad | 0.83 | 0.42 | 0.56 | 12 |
| Crisis | 0.36 | 0.80 | 0.50 | 5 |
| **Accuracy** | | | **0.53** | 17 |

### 7.2 Población · resumen global (notebooks 03–05, nivel día)

| Experimento | Etiqueta | Estudiantes | Días | Matriz `[[TN, FP], [FN, TP]]` | Accuracy | Precisión (+) | Recall (+) | F1 (+) |
|---|---|---|---|---|---|---|---|---|
| 03 · Estrés | estrés > P70 | 8 | 160 | `[[60, 52], [6, 42]]` | 0.64 | 0.45 | 0.88 | 0.59 |
| 04 · Carga mental | máx(estrés, ansiedad) > P70 | 8 | 160 | `[[37, 39], [7, 77]]` | 0.71 | 0.66 | 0.92 | 0.77 |
| 05 · Ansiedad | ansiedad > P70 | 9 | 174 | `[[31, 76], [6, 61]]` | 0.53 | 0.45 | 0.91 | 0.60 |

### 7.3 Población · F1 por estudiante

| Estudiante | Días evaluados | 03 Estrés | 04 Carga mental | 05 Ansiedad |
|---|---|---|---|---|
| 1 | 18 | 0.50 | 0.60 | 0.60 |
| 2 | 14 | error* | error* | 0.56 |
| 3 | — | sin HR | sin HR | sin HR |
| 4 | 5 | 0.57 | 0.57 | 0.57 |
| 5 | 31 | 0.71 | 0.74 | 0.71 |
| 6 | 11 | 0.82 | 0.82 | 0.82 |
| 7 | 31 | 0.50 | 0.50 | 0.40 |
| 8 | 25 | 0.24 | 0.81 | 0.50 |
| 9 | 19 | 0.57 | 0.91 | 0.64 |
| 10 | 20 | 0.54 | 0.77 | 0.52 |

\* `Columns must be same length as key`, pendiente de depurar.

### 7.4 Patrones de datos faltantes (notebook 06, estudiantes 20–35)

| Patrón (1 = falta) | ρ Ansiedad | ρ Estrés |
|---|---|---|
| BPM0_RMSSD0_SPO20 | −0.103 | −0.033 |
| BPM0_RMSSD0_SPO21 | −0.056 | −0.034 |
| BPM0_RMSSD1_SPO20 | −0.048 | −0.030 |
| BPM0_RMSSD1_SPO21 | 0.015 | 0.025 |
| BPM1_RMSSD0_SPO20 | −0.128 | −0.007 |
| BPM1_RMSSD0_SPO21 | −0.078 | 0.089 |
| BPM1_RMSSD1_SPO20 | −0.094 | −0.006 |
| BPM1_RMSSD1_SPO21 | 0.087 | 0.047 |

Todas las correlaciones son **muy débiles** (|ρ| < 0.13). La falta de datos no parece explicar el estado emocional.

### Lectura general

- El sistema **detecta casi todos los días de crisis**, con recall de 0.88–0.92 en la población.
- A cambio genera **muchas falsas alarmas**: la precisión es baja y el recall de los días de paz va de 0.29 a 0.54.
- La etiqueta combinada (**carga mental**) es la que mejor funciona.

---

## 8. Dashboard

`dashboard/app_dashboard_smieae.py` lee `SMIEAE_Reporte_Completo.csv` y muestra seis pestañas:

1. **Resumen global:** bloques normales contra anómalos por mes.
2. **Desglose por días:** barras de bloques junto con las líneas de estrés y ansiedad reportados, el umbral de paz y las semanas del calendario académico.
3. **Análisis de anomalías:** comparador de dos días por su carga acumulada en 60 minutos.
4. **Días normales vs anómalos:** tabla según un percentil de alarma.
5. **Semáforo (picos):** rojo si hay ≥7 bloques anómalos en 60 minutos y amarillo si hay ≥4.
6. **Estrés crónico (volumen):** total de bloques anómalos del día, con cortes en P75 y P50.

En la barra lateral se ajustan el percentil de corte de EOTE (por defecto P90) y el percentil de alarma diaria (por defecto P75).

```bash
streamlit run dashboard/app_dashboard_smieae.py --server.port 8506
```

---

## 9. Instalación y ejecución

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit pytz jupyter
```

**Antes de correr los notebooks:**

1. Coloca el módulo **`EOTE/`** en la raíz del repo. Los notebooks lo importan con `sys.path.append(".")`. Si los mueves a `notebooks/`, cambia la línea a `sys.path.append("..")`.
2. Coloca el dataset en `data/SSAQS/` y reemplaza las rutas absolutas (`C:\Users\esteb\...`) por una variable relativa, por ejemplo `BASE = Path("../data/SSAQS")`.
3. Ejecuta cada notebook **de arriba hacia abajo** (*Restart & Run All*).

`.gitignore` recomendado. Son datos de salud: no los subas.

```
data/
outputs/
*.csv
.venv/
__pycache__/
.ipynb_checkpoints/
```

---

## 10. Limitaciones y pendientes

Estos puntos conviene revisarlos antes de reportar resultados finales.

**Validez de la evaluación**
- [ ] **El umbral diario se elige sobre el propio conjunto de prueba** (notebooks 03–05): se busca el que maximiza F1 en el 30 % de prueba, así que las métricas son optimistas. Lo correcto es elegirlo en una validación dentro del 70 %.
- [ ] **El split 70/30 de la población es por bloques** (`iloc`), no por días. Un mismo día puede quedar partido entre entrenamiento y prueba. El notebook 02 sí separa por días completos.
- [ ] **Los percentiles P25/P70 de la población se calculan con todo el periodo**, incluida la prueba. El notebook 02 los calcula solo con el pasado.
- [ ] El notebook 01 evalúa sobre los mismos datos con los que entrenó, así que solo sirve como exploración.

**Agregación del puntaje**
- [ ] Los puntajes de EOTE observados son negativos (el P90 es ≈ −4.9). Al **sumarlos por día**, los días con más bloques registrados obtienen totales más bajos, y el umbral puede terminar reflejando cuánto se usó el reloj más que la anomalía. Conviene probar la media o la proporción de bloques anómalos.

**Consistencia del código**
- [ ] Los mensajes dicen "Crisis (P75)" pero el código usa P70.
- [ ] Las gráficas de 03 y 05 dicen "N=9" y "Días de Crisis" aunque cambie la población o la etiqueta.
- [ ] En el notebook 02 el título dice "8 Pilares", pero se calculan 5.
- [ ] El notebook 01 se ejecutó fuera de orden. Las celdas de umbral (P70 de estrés, P50 de ansiedad) corrieron después de la que etiqueta normalidad, y los comentarios dicen "Percentil 25". Además, el dashboard dibuja la línea de paz en P25 aunque el modelo se entrenó con otro umbral.
- [ ] La tabla de importancias "promediadas" del notebook 01 usa valores copiados a mano de otra ejecución y promedia vectores de distinto tamaño. El resultado es casi uniforme y no es informativo.
- [ ] Hay que depurar el error del estudiante 2 en 03 y 04. El estudiante 3 no tiene datos de frecuencia cardiaca.
- [ ] La columna se llama `HRV_RMSDD` (debería ser RMSSD). Se mantiene el nombre por compatibilidad.

**Alcance**
- [ ] El notebook 06 junta los días de varios estudiantes sin normalizar por persona y no incluye pruebas de significancia.
- [ ] La conversión de zona horaria usa un desfase fijo de −6 h.
- [ ] La muestra es pequeña: entre 5 y 31 días de prueba por estudiante.
