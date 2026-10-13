# SMIEAE

Detección de estrés y ansiedad en estudiantes universitarios a partir de anomalías en datos fisiológicos de Fitbit.

La idea del proyecto es entrenar un modelo de detección de anomalías (EOTE) únicamente con los días en que cada estudiante reportó estar tranquilo. Después se revisa si los días que el modelo marca como anómalos coinciden con los días en que el estudiante reportó estrés o ansiedad alta en su encuesta diaria.

Autor: Javier Esteban Torres Sossa

## Estructura del repositorio

```
SMIEAE/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_exploracion_estudiante1_eote.py
│   ├── 02_estudiante1_split7030_random_forest.py
│   ├── 03_poblacion_eote_estres.py
│   ├── 04_poblacion_eote_carga_mental.py
│   ├── 05_poblacion_eote_ansiedad.py
│   └── 06_correlacion_patrones_datos_faltantes.py
├── dashboard/
│   └── app_dashboard_smieae.py
├── docs/
│   └── presentacion_deteccion_anomalias_2026-06-28.pdf
└── EOTE/
```

Los datos de los estudiantes no se incluyen en el repositorio.

## Datos

Se usa el dataset SSAQS. Cada estudiante tiene una carpeta con estos archivos:

```
<id_estudiante>/
├── steps.csv
├── oxygen.csv
├── hrv.csv
├── daily_questions.csv
└── Global Export Data/
    └── *heart_rate*.json
```

Las marcas de tiempo vienen en UTC y se pasan a hora local restando 6 horas.

La encuesta diaria (`daily_questions.csv`) tiene los puntajes de `stress` y `anxiety` en una escala de 0 a 100.

## Preprocesamiento

Todo se agrupa en bloques de 5 minutos. De cada bloque se calculan 13 variables:

| Variable | Descripción |
|---|---|
| HR_MEAN | Promedio de BPM. Si falta un solo bloque se interpola |
| HR_STD | Desviación estándar de BPM |
| first_diff | Diferencia de HR_MEAN con el bloque anterior |
| HR_MEAN_5 | Media móvil de HR_MEAN en 5 bloques |
| HR_STD_5 | Desviación móvil de HR_MEAN en 5 bloques |
| HRV_RMSDD, HRV_RMSDD_STD | Media y desviación móvil de RMSSD (se quitan valores ≤ 0 y > 200) |
| HRV_LF, HRV_LF_STD | Media y desviación móvil de baja frecuencia |
| HRV_HF, HRV_HF_STD | Media y desviación móvil de alta frecuencia |
| steps | Suma de pasos |
| SPO2_MEAN | Promedio de SpO2 (se quitan valores menores a 80) |

Después de unir los sensores con la encuesta se hacen dos filtros:

1. Se eliminan los bloques donde no hay ningún dato de ritmo cardiaco ni de HRV. Sin este filtro el modelo marcaba como anomalía bloques que solo tenían pasos.
2. Solo se conservan los días con al menos 144 bloques, que equivalen a unas 12 horas de uso del reloj.

Para el estudiante 1 los registros pasaron de 30,343 a 13,461 con el primer filtro, y a 12,406 con el segundo.

| Variable | Faltantes antes | Faltantes después |
|---|---|---|
| HR_MEAN | 57.93 % | 2.90 % |
| HRV_RMSSD | 90.51 % | 80.41 % |
| SPO2_MEAN | 89.24 % | 78.02 % |
| steps | 0 % | 0 % |

## Criterio de paz y crisis

Los umbrales se calculan por estudiante con el valor máximo diario de su encuesta:

- Día de paz: puntaje menor o igual al percentil 25. Con estos días se entrena el modelo.
- Día de crisis: puntaje mayor al percentil 70.

## Experimentos

| Script | Qué hace |
|---|---|
| `01_exploracion_estudiante1_eote.py` | Exploración del estudiante 1. Limpieza de datos, entrenamiento de EOTE, importancia de variables, ejemplo de interpretación de una predicción y exportación de rankings de anomalías. También lanza el dashboard. |
| `02_estudiante1_split7030_random_forest.py` | Estudiante 1 con división cronológica 70/30 por días completos. Se calcula una carga emocional que combina estrés y ansiedad. Con los puntajes de EOTE se construyen 5 variables por día y se entrena un Random Forest para separar días normales de días de crisis. |
| `03_poblacion_eote_estres.py` | Un modelo EOTE por estudiante (1 al 10). Se entrena con días de paz según estrés y la etiqueta es estrés mayor al P70. |
| `04_poblacion_eote_carga_mental.py` | Igual que el 03, pero la etiqueta es el máximo entre estrés y ansiedad. |
| `05_poblacion_eote_ansiedad.py` | Igual que el 03, pero todo con ansiedad. |
| `06_correlacion_patrones_datos_faltantes.py` | Revisa si los huecos en los sensores tienen relación con el estrés o la ansiedad. Usa los estudiantes 20 al 35. |

En los scripts 03, 04 y 05 cada día se clasifica sumando los puntajes de anomalía de sus bloques. Esa suma se compara contra un umbral que se elige probando 100 valores y quedándose con el que da el mejor F1.

En el script 02 las variables diarias son:

- `Carga_Total_Estres`: suma de puntajes de los bloques anómalos.
- `Severidad_Promedio`: promedio de puntajes de los bloques anómalos.
- `Tren_Critico_30m`: la suma más alta de 6 bloques seguidos.
- `Indice_Paz_Porcentual`: porcentaje de bloques normales del día.
- `Racha_Paz_Maxima_Minutos`: la racha más larga de bloques normales seguidos, en minutos.

Un bloque se considera anómalo si su puntaje supera el percentil 90 de los puntajes del periodo de entrenamiento.

## Resultados

### Población (scripts 03, 04 y 05)

| Etiqueta | Estudiantes | Días | Accuracy | Precisión crisis | Recall crisis | F1 crisis |
|---|---|---|---|---|---|---|
| Estrés | 8 | 160 | 0.64 | 0.45 | 0.88 | 0.59 |
| Estrés o ansiedad | 8 | 160 | 0.71 | 0.66 | 0.92 | 0.77 |
| Ansiedad | 9 | 174 | 0.53 | 0.45 | 0.91 | 0.60 |

Matrices de confusión (filas: real, columnas: predicción; orden paz, crisis):

- Estrés: `[[60, 52], [6, 42]]`
- Estrés o ansiedad: `[[37, 39], [7, 77]]`
- Ansiedad: `[[31, 76], [6, 61]]`

F1 por estudiante:

| Estudiante | Días | Estrés | Estrés o ansiedad | Ansiedad |
|---|---|---|---|---|
| 1 | 18 | 0.50 | 0.60 | 0.60 |
| 2 | 14 | error | error | 0.56 |
| 3 | - | sin datos de HR | sin datos de HR | sin datos de HR |
| 4 | 5 | 0.57 | 0.57 | 0.57 |
| 5 | 31 | 0.71 | 0.74 | 0.71 |
| 6 | 11 | 0.82 | 0.82 | 0.82 |
| 7 | 31 | 0.50 | 0.50 | 0.40 |
| 8 | 25 | 0.24 | 0.81 | 0.50 |
| 9 | 19 | 0.57 | 0.91 | 0.64 |
| 10 | 20 | 0.54 | 0.77 | 0.52 |

En general el modelo encuentra casi todos los días de crisis, pero también marca muchos días tranquilos como crisis. La etiqueta que combina estrés y ansiedad es la que mejor resultado da.

### Estudiante 1 con Random Forest (script 02)

La división quedó en 54 días válidos con corte el 2025-04-20. En el entrenamiento hubo 26 días normales y 10 de crisis. En la prueba:

| Clase | Precisión | Recall | F1 | Días |
|---|---|---|---|---|
| Normalidad | 0.83 | 0.42 | 0.56 | 12 |
| Crisis | 0.36 | 0.80 | 0.50 | 5 |

Accuracy: 0.53

### Datos faltantes (script 06)

Correlación de Spearman entre el porcentaje diario de cada patrón de datos faltantes y la encuesta (1 = falta el sensor):

| Patrón | Ansiedad | Estrés |
|---|---|---|
| BPM0_RMSSD0_SPO20 | -0.103 | -0.033 |
| BPM0_RMSSD0_SPO21 | -0.056 | -0.034 |
| BPM0_RMSSD1_SPO20 | -0.048 | -0.030 |
| BPM0_RMSSD1_SPO21 | 0.015 | 0.025 |
| BPM1_RMSSD0_SPO20 | -0.128 | -0.007 |
| BPM1_RMSSD0_SPO21 | -0.078 | 0.089 |
| BPM1_RMSSD1_SPO20 | -0.094 | -0.006 |
| BPM1_RMSSD1_SPO21 | 0.087 | 0.047 |

Todas las correlaciones son muy bajas, así que la falta de datos no parece estar relacionada con el estado del estudiante.

### Versión anterior (presentación)

En `docs/` está la presentación del 28 de junio de 2026. Esa versión usaba también el nivel de actividad del Fitbit, separaba con el percentil 70 y evaluaba bloque por bloque en lugar de día por día. También se probó un modelo por cada parte del día:

| Modelo | Exactitud | Sensibilidad | Precisión | F1 |
|---|---|---|---|---|
| Todo el día | 63.36 % | 39.3 % | 39.3 % | 39.3 % |
| Mañana | 68.8 % | 47.9 % | 45.4 % | 46.6 % |
| Tarde | 65.6 % | 42.9 % | 44.2 % | 43.6 % |
| Noche | 76.5 % | 41.3 % | 56.0 % | 47.5 % |
| Madrugada | 66.66 % | 44.6 % | 45.9 % | 45.2 % |

## Dashboard

El dashboard está hecho en Streamlit y lee el archivo `SMIEAE_Reporte_Completo.csv` que genera el script 01. Tiene seis pestañas:

1. Resumen global por mes.
2. Desglose por día, con las líneas de estrés y ansiedad y las semanas del calendario escolar.
3. Comparación de dos días con anomalías.
4. Tabla de días normales y anómalos.
5. Semáforo según la cantidad de bloques anómalos en una hora.
6. Volumen total de anomalías por día.

Para correrlo:

```
streamlit run dashboard/app_dashboard_smieae.py --server.port 8506
```

## Cómo correr el proyecto

Instalar dependencias:

```
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit pytz
```

Antes de correr los scripts:

1. La carpeta `EOTE/` tiene que estar en la raíz del proyecto. Los scripts la importan con `sys.path.append(".")`, así que hay que ejecutarlos desde la raíz.
2. Cambiar las rutas de los datos en cada script (`base_path`, `ruta_base` y las rutas de cada archivo) por la ubicación del dataset en tu computadora.

## Pendientes

- En los scripts 03 a 05 el umbral diario se elige con los mismos datos de prueba, por lo que los resultados pueden estar sobreestimados. Falta separar un conjunto de validación.
- En esos mismos scripts la división 70/30 es por bloques y no por días, así que un día puede quedar partido entre entrenamiento y prueba.
- Los percentiles de paz y crisis de los scripts 03 a 05 se calculan con todo el periodo. En el script 02 ya se calculan solo con el periodo de entrenamiento.
- Los puntajes de EOTE salen negativos, entonces al sumarlos los días con más registros quedan con valores más bajos. Vale la pena probar con el promedio o con la proporción de bloques anómalos.
- Revisar el error del estudiante 2 en los scripts 03 y 04 ("Columns must be same length as key").
- Algunos textos no coinciden con el código: los mensajes dicen P75 pero se usa P70, las gráficas dicen N=9 en todos los casos y el script 02 menciona 8 pilares cuando son 5.
- El script 01 se ejecutó con las celdas fuera de orden. Hay que correrlo completo de arriba hacia abajo para confirmar los resultados.
- La columna `HRV_RMSDD` debería llamarse `HRV_RMSSD`.
