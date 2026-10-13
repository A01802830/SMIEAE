"""
01_exploracion_estudiante1_eote.py

Exploración del estudiante 1: limpieza, EOTE sobre todo el semestre,
interpretabilidad, exportación de rankings y lanzamiento del dashboard.

Proyecto SMIEAE · Detección de estrés y ansiedad mediante anomalías fisiológicas.
Autor: Javier Esteban Torres Sossa
"""


# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
import sys
import warnings
import pytz
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# %%
# 2. PASOS
ruta_steps = r'C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)\1\steps.csv'
df_steps = pd.read_csv(ruta_steps)
df_steps['timestamp'] = pd.to_datetime(df_steps['timestamp'], utc=True)
df_steps = df_steps.set_index('timestamp')
df_steps.index = (df_steps.index - pd.Timedelta(hours=6)).tz_localize(None)
steps_5min = df_steps['steps'].resample('5min').sum()


# %%
# 3. OXÍGENO
ruta_oxy = r'C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)\1\oxygen.csv'
df_oxygen = pd.read_csv(ruta_oxy)
df_oxygen = df_oxygen[df_oxygen['value'] >= 80] 
df_oxygen['timestamp'] = pd.to_datetime(df_oxygen['timestamp'], utc=True)
df_oxygen = df_oxygen.set_index('timestamp')
df_oxygen.index = (df_oxygen.index - pd.Timedelta(hours=6)).tz_localize(None)

oxygen_5min = df_oxygen['value'].resample('5min').agg(['mean'])
oxygen_5min.columns = ['SPO2_MEAN']


# %%
# 4. HRV
ruta_hrv = r'C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)\1\hrv.csv'
df_hrv = pd.read_csv(ruta_hrv)
df_hrv['timestamp'] = pd.to_datetime(df_hrv['timestamp'], utc=True)
df_hrv = df_hrv.set_index('timestamp')
df_hrv.index = (df_hrv.index - pd.Timedelta(hours=6)).tz_localize(None)

df_hrv.loc[df_hrv['rmssd'] > 200, 'rmssd'] = np.nan
df_hrv.loc[df_hrv['rmssd'] <= 0, 'rmssd'] = np.nan
df_hrv.loc[df_hrv['low_frequency'] < 0, 'low_frequency'] = np.nan
df_hrv.loc[df_hrv['high_frequency'] < 0, 'high_frequency'] = np.nan

hrv_stats = df_hrv[['rmssd', 'low_frequency', 'high_frequency']].rolling(window=3).agg(['mean', 'std'])
hrv_stats.columns = ['HRV_RMSDD', 'HRV_RMSDD_STD','HRV_LF','HRV_LF_STD','HRV_HF','HRV_HF_STD']

# 5. RITMO CARDÍACO
ruta_carpeta = r"C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)\1\Global Export Data"
busqueda = os.path.join(ruta_carpeta,"*heart_rate*.json")
archivos_json = glob.glob(busqueda)

datos_completos = []
for archivo in archivos_json:
    with open(archivo, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entrada in data:
            datos_completos.append({
                'dateTime': entrada['dateTime'],
                'bpm': entrada['value']['bpm']
            })

df_heart_rate = pd.DataFrame(datos_completos)
df_heart_rate['dateTime'] = pd.to_datetime(df_heart_rate['dateTime'], format='%m/%d/%y %H:%M:%S')
df_heart_rate = df_heart_rate.set_index('dateTime')
df_heart_rate.index = (df_heart_rate.index.tz_localize('UTC') - pd.Timedelta(hours=6)).tz_localize(None)
# 1. Agrupación base a 5 minutos
heart_rate_5min = df_heart_rate['bpm'].resample('5min').agg(['mean','std'])
heart_rate_5min.columns = ['HR_MEAN','HR_STD']

# 2. INTERPOLACIÓN LIGERA (El salvavidas de micro-huecos)
# Si solo falta 1 bloque de 5 minutos, lo unimos con una línea suave. 
# Si faltan 3 horas, limit=1 detiene la invención de datos y los deja como NaN.
heart_rate_5min['HR_MEAN'] = heart_rate_5min['HR_MEAN'].interpolate(method='linear', limit=1)

# 3. MATEMÁTICAS CON NANS (El truco maestro)
# Calculamos las derivadas y ventanas sobre todo el dataset.
# Hacer esto ANTES de borrar los NaNs garantiza que el modelo no calcule un .diff()
# falso saltándose 3 horas de tiempo.
heart_rate_5min['first_diff'] = heart_rate_5min['HR_MEAN'].diff()
heart_rate_5min['HR_MEAN_5'] = heart_rate_5min['HR_MEAN'].rolling(window=5, min_periods=1).mean()
heart_rate_5min['HR_STD_5'] = heart_rate_5min['HR_MEAN'].rolling(window=5, min_periods=1).std()

# 4. LA PODA ESTRICTA (Limpieza)
# Borramos cualquier fila que se haya quedado como NaN (los huecos de horas).
# Además, borramos donde first_diff sea NaN (el momento exacto en que se reconectó el reloj).
heart_rate_5min = heart_rate_5min.dropna(subset=['HR_MEAN', 'first_diff'])

# Rellenamos los NaNs de la desviación estándar (si hubo ventanas de 1 solo latido)
heart_rate_5min['HR_STD_5'] = heart_rate_5min['HR_STD_5'].fillna(0)
heart_rate_5min['HR_STD'] = heart_rate_5min['HR_STD'].fillna(0)


# %%
# Fusión del Dataset Maestro
dataset_maestro = pd.concat([heart_rate_5min, hrv_stats, steps_5min, oxygen_5min], axis=1)
dataset_maestro.columns = ['HR_MEAN', 'HR_STD', 'first_diff','HR_MEAN_5','HR_STD_5', 'HRV_RMSDD', 'HRV_RMSDD_STD', 'HRV_LF', 'HRV_LF_STD', 'HRV_HF', 'HRV_HF_STD', 'steps', 'SPO2_MEAN']
ruta_daily = r'C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)\1\daily_questions.csv'
df_daily = pd.read_csv(ruta_daily)
df_daily['timestamp'] = pd.to_datetime(df_daily['timeStampStart'], unit='s', utc=True)
df_daily['timestamp'] = (df_daily['timestamp'] - pd.Timedelta(hours=6)).dt.tz_localize(None)

df_daily['fecha'] = df_daily['timestamp'].dt.date
daily_summary = df_daily.groupby('fecha')[['stress', 'anxiety']].mean()

dataset_maestro['fecha'] = dataset_maestro.index.date
dataset_maestro = dataset_maestro.join(daily_summary, on='fecha')
dataset_maestro = dataset_maestro.drop(columns=['fecha'])


# %%
print(len(dataset_maestro))


# %%
#Datos faltantes
HR_MEAN = dataset_maestro['HR_MEAN'].isna().sum() / len(dataset_maestro)
HRV_RMSSD = dataset_maestro['HRV_RMSDD'].isna().sum() / len(dataset_maestro)
SPO2_MEAN = dataset_maestro['SPO2_MEAN'].isna().sum() / len(dataset_maestro)
STEPS = dataset_maestro['steps'].isna().sum() / len(dataset_maestro)



print(f"Proporción de datos faltantes en HR_MEAN: {HR_MEAN:.2%} ")
print(f"Proporción de datos faltantes en HRV_RMSSD: {HRV_RMSSD:.2%} ")
print(f"Proporción de datos faltantes en SPO2_MEAN: {SPO2_MEAN:.2%} ")
print(f"Proporción de datos faltantes en STEPS: {STEPS:.2%} ")


# %%
# Limpieza de Fantasmas
print(f"Cantidad de registros antes de la limpieza de fantasmas: {len(dataset_maestro)}")
variables_fisiologicas = ['HR_MEAN', 'HR_STD', 'first_diff', 'HRV_RMSDD', 'HRV_RMSDD_STD', 'HRV_LF', 'HRV_LF_STD', 'HRV_HF', 'HRV_HF_STD']
dataset_maestro = dataset_maestro.dropna(subset=variables_fisiologicas, how='all')
print(f"Cantidad de registros después de la limpieza de fantasmas: {len(dataset_maestro)}")
# Filtro de Adherencia (Limpieza de días vacíos)
print(f"Cantidad de registros antes del filtro de adherencia: {len(dataset_maestro)}")
conteo_diario = dataset_maestro.groupby(dataset_maestro.index.date).size()
umbral_adherencia = 144
dias_validos = conteo_diario[conteo_diario >= umbral_adherencia].index
dataset_maestro = dataset_maestro[np.isin(dataset_maestro.index.date, dias_validos)]
print(f"Cantidad de registros después del filtro de adherencia: {len(dataset_maestro)}")


# %%
#Datos faltantes
HR_MEAN = dataset_maestro['HR_MEAN'].isna().sum() / len(dataset_maestro)
HRV_RMSSD = dataset_maestro['HRV_RMSDD'].isna().sum() / len(dataset_maestro)
SPO2_MEAN = dataset_maestro['SPO2_MEAN'].isna().sum() / len(dataset_maestro)
STEPS = dataset_maestro['steps'].isna().sum() / len(dataset_maestro)



print(f"Proporción de datos faltantes en HR_MEAN: {HR_MEAN:.2%} ")
print(f"Proporción de datos faltantes en HRV_RMSSD: {HRV_RMSSD:.2%} ")
print(f"Proporción de datos faltantes en SPO2_MEAN: {SPO2_MEAN:.2%} ")
print(f"Proporción de datos faltantes en STEPS: {STEPS:.2%} ")


# %%
# Definir el Umbral de Paz (Percentil 25)
puntajes_diarios = dataset_maestro.groupby(dataset_maestro.index.date)['stress'].first().dropna()
umbral_paz_personal = np.percentile(puntajes_diarios, 70)

print(f"El Umbral de Paz (Percentil 25) es: {umbral_paz_personal:.1f}")
dias_normales = puntajes_diarios[puntajes_diarios <= umbral_paz_personal].index
print(f"Días de paz total para entrenar: {len(dias_normales)} días.")


# %%
# Definir el Umbral de Paz (Percentil 25)
puntajes_diarios = dataset_maestro.groupby(dataset_maestro.index.date)['anxiety'].first().dropna()
umbral_paz_personal = np.percentile(puntajes_diarios, 50)

print(f"El Umbral de Paz (Percentil 25) es: {umbral_paz_personal:.1f}")
dias_normales = puntajes_diarios[puntajes_diarios <= umbral_paz_personal].index
print(f"Días de paz total para entrenar: {len(dias_normales)} días.")


# %%
sys.path.append(".")
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder

# 1. Definir qué es normalidad basándonos en la encuesta
def normal(valor):
    if valor['stress'] <= umbral_paz_personal:
        return 'normal'
    else:
        return 'anomalo'
    

dataset_maestro['estado'] = dataset_maestro.apply(normal, axis=1)


# %%
df_entrenamiento =dataset_maestro[dataset_maestro['estado'] == 'normal']
#Datos faltantes
HR_MEAN = df_entrenamiento['HR_MEAN'].isna().sum() / len(df_entrenamiento)
HRV_RMSSD = df_entrenamiento['HRV_RMSDD'].isna().sum() / len(df_entrenamiento)
SPO2_MEAN = df_entrenamiento['SPO2_MEAN'].isna().sum() / len(df_entrenamiento)
STEPS = df_entrenamiento['steps'].isna().sum() / len(df_entrenamiento)
Stress = df_entrenamiento['stress'].isna().sum() / len(df_entrenamiento)


print(f"Proporción de datos faltantes en HR_MEAN: {HR_MEAN:.2%} ")
print(f"Proporción de datos faltantes en HRV_RMSSD: {HRV_RMSSD:.2%} ")
print(f"Proporción de datos faltantes en SPO2_MEAN: {SPO2_MEAN:.2%} ")
print(f"Proporción de datos faltantes en STEPS: {STEPS:.2%} ")
print(f"Proporción de datos faltantes en Stress: {Stress:.2%} ")


# %%
print(dataset_maestro.head())


# %%
# 2. Extraer SOLO la normalidad pura para entrenar a EOTE
normalidad_pura = dataset_maestro[dataset_maestro['estado'] == 'normal'].copy()
columnas_ignorar = ['stress', 'anxiety', 'estado']

x_train = normalidad_pura.drop(columns=columnas_ignorar, errors='ignore').reset_index(drop=True)
y_train = pd.DataFrame({'class': ['normal'] * len(x_train)}, index=x_train.index)


print(f"🧠 Entrenando EOTE con {len(x_train)} bloques puros de los Días de Paz...")
director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
eote = director.get_eote()
eote.train(x_train, y_train)

# 3. Clasificar (Predecir) sobre TODO el semestre completo
print(f"🔍 Evaluando el semestre completo para encontrar las Incongruencias con {len(dataset_maestro)} bloques...")
x_test_global = dataset_maestro.drop(columns=columnas_ignorar, errors='ignore')

dataset_maestro['anomaly_score'] = eote.classify(x_test_global)

# Etiquetar para el dashboard
dataset_maestro['Fase_Experimento'] = 'Evaluación Global contra Días de Paz'

dataset_maestro.to_csv("SMIEAE_Reporte_Completo.csv") 
print("\n✅ Modelo entrenado y datos exportados a SMIEAE_Reporte_Completo.csv")


# %%
print("📊 Importancia de variables para la detección de estrés:\n")
for feature_tree in eote._EOTE__feature_trees:
    # Preguntamos si tiene classifier (categórico) o regressor (numérico)
    if hasattr(feature_tree, 'classifier') and feature_tree.classifier is not None:
        importancia = feature_tree.classifier.feature_importances_
    elif hasattr(feature_tree, 'regressor') and feature_tree.regressor is not None:
        importancia = feature_tree.regressor.feature_importances_
    else:
        importancia = "No disponible"
    
    print(f"Variable: {feature_tree.feature.name():<20} | Importancia: {importancia}")


# %%
import pandas as pd
import numpy as np

# Datos extraídos de tu impresión
data = {
    "HR_MEAN": [0.114, 0.014, 0.021, 0.046, 0.019, 0.033, 0.047, 0.566, 0.044, 0.061, 0.028],
    "HR_STD": [0.088, 0.155, 0.023, 0.018, 0.160, 0.007, 0.037, 0.029, 0.090, 0.328, 0.059],
    "first_diff": [0.076, 0.070, 0.102, 0.147, 0.018, 0.148, 0.146, 0.054, 0.029, 0.028, 0.113, 0.063],
    "HR_MEAN_5": [0.909, 0.001, 0.028, 0.017, 0.001, 0.000, 0.004, 0.001, 0.026, 0.000, 0.000, 0.006],
    "HR_STD_5": [0.048, 0.000, 0.219, 0.043, 0.021, 0.125, 0.032, 0.300, 0.021, 0.107, 0.000, 0.079],
    "HRV_RMSDD": [0.018, 0.011, 0.008, 0.017, 0.019, 0.019, 0.020, 0.841, 0.030, 0.002, 0.009],
    "HRV_RMSDD_STD": [0.028, 0.039, 0.019, 0.020, 0.024, 0.171, 0.020, 0.047, 0.092, 0.505, 0.009, 0.020],
    "HRV_LF": [0.014, 0.017, 0.020, 0.025, 0.027, 0.042, 0.069, 0.687, 0.063, 0.004, 0.026],
    "HRV_LF_STD": [0.013, 0.006, 0.017, 0.022, 0.047, 0.019, 0.035, 0.259, 0.016, 0.532, 0.008, 0.021],
    "HRV_HF": [0.000, 0.002, 0.067, 0.015, 0.001, 0.083, 0.012, 0.798, 0.015, 0.000, 0.002],
    "HRV_HF_STD": [0.002, 0.000, 0.011, 0.018, 0.001, 0.005, 0.062, 0.002, 0.177, 0.715, 0.000, 0.000],
    "steps": [0.318, 0.076, 0.038, 0.023, 0.037, 0.019, 0.246, 0.021, 0.067, 0.037, 0.059, 0.052],
    "SPO2_MEAN": [0.044, 0.019, 0.006, 0.013, 0.062, 0.090, 0.072, 0.042, 0.076, 0.515, 0.054, 0.002]
}

# Promediamos para obtener un solo valor de importancia por variable
importancia_final = {k: np.mean(v) for k, v in data.items()}
df_final = pd.DataFrame.from_dict(importancia_final, orient='index', columns=['Importancia'])
df_final = df_final.sort_values(by='Importancia', ascending=False)

print(df_final)


# %%
eote.classify_and_interpret(x_test_global.iloc[22])


# %%
# El dashboard de Streamlit (antes app5.py, generado aquí con %%writefile)
# ahora vive en: dashboard/app_dashboard_smieae.py


# %%
import os
import subprocess
import time

puerto = 8506
archivo_app = "dashboard/app_dashboard_smieae.py"

print("🔍 Revisando si hay dashboards anteriores corriendo...")

# 1. Limpieza automática (Exclusivo para tu entorno Windows)
try:
    resultado = subprocess.check_output(f"netstat -ano | findstr :{puerto}", shell=True).decode()
    for linea in resultado.strip().split('\n'):
        if "LISTENING" in linea:
            pid = linea.strip().split()[-1]
            os.system(f"taskkill /PID {pid} /F >nul 2>&1")
            print(f"🧹 Dashboard anterior cerrado exitosamente (PID: {pid}).")
except Exception:
    print("✅ El puerto está limpio y listo.")

# 2. Lanzar la nueva app de Streamlit en segundo plano
print(f"🚀 Levantando {archivo_app}...")
comando = f"python -m streamlit run {archivo_app} --server.port {puerto} --server.headless true"
proceso = subprocess.Popen(comando, shell=True)

# 3. Esperar 3 segundos para que el servidor despierte y mostrar el link
time.sleep(3)
print("\n" + "="*50)
print(f"🎉 ¡DASHBOARD CORRIENDO!")
print(f"🌐 Haz clic aquí para abrirlo: http://localhost:{puerto}")
print("="*50)
print("💡 Nota: Si haces cambios, edita dashboard/app_dashboard_smieae.py y vuelve a ejecutar este bloque.")


# %%
import pandas as pd

# Cargamos el dataset completo directamente
df_analisis = pd.read_csv("SMIEAE_Reporte_Completo.csv", index_col=0, parse_dates=True)
df_analisis.index = pd.to_datetime(df_analisis.index, errors='coerce')
df_analisis = df_analisis.sort_index()

# 1. Agrupamos por día y extraemos el valor de la encuesta para el estudiante
puntajes_diarios = df_analisis.groupby(df_analisis.index.date)['stress'].first()

# 2. Limpiamos los vacíos (días sin reporte)
puntajes_diarios = puntajes_diarios.dropna()

# 3. Ordenamos de mayor a menor nivel de estrés
puntajes_diarios = puntajes_diarios.sort_values(ascending=False)

# 4. Imprimimos el Top 20
print("🔝 TOP 20 DÍAS CON MAYOR ESTRÉS REPORTADO:")
print(puntajes_diarios.head(20))


# %%
import pandas as pd

# 1. Cargar el dataset
df_anomalias = pd.read_csv("SMIEAE_Reporte_Completo.csv", index_col=0, parse_dates=True)
df_anomalias.index = pd.to_datetime(df_anomalias.index, errors='coerce')

# 2. Identificar la columna correcta
col_maestra = 'anomaly_scores' if 'anomaly_scores' in df_anomalias.columns else 'anomaly_score'

# 3. Limpieza de strings
df_anomalias[col_maestra] = df_anomalias[col_maestra].astype(str).str.replace(r'np\.float64', '', regex=True)
df_anomalias[col_maestra] = df_anomalias[col_maestra].str.replace(r'[\[\]\(\)]', '', regex=True)
df_anomalias[col_maestra] = pd.to_numeric(df_anomalias[col_maestra], errors='coerce')

# 4. Ordenar TODAS las instancias de mayor a menor (sin usar .head())
todas_las_anomalias = df_anomalias.dropna(subset=[col_maestra])[col_maestra].sort_values(ascending=False)

# ==========================================
# OPCIÓN A: Exportar a un archivo nuevo (RECOMENDADO)
# ==========================================
todas_las_anomalias.to_csv("Ranking_Completo_Anomalias.csv", header=True)
print("✅ ¡Listo! Revisa tu carpeta, se creó el archivo 'Ranking_Completo_Anomalias.csv' con todas las instancias.")

# ==========================================
# OPCIÓN B: Forzar a Pandas a imprimir todo en pantalla
# ==========================================
# Descomenta las siguientes dos líneas si realmente quieres que la libreta imprima todas las filas
# pd.set_option('display.max_rows', None) 
# print(todas_las_anomalias)


# %%
import pandas as pd

# 1. Cargar el dataset completo
df_anomalias = pd.read_csv("SMIEAE_Reporte_Completo.csv", index_col=0, parse_dates=True)
df_anomalias.index = pd.to_datetime(df_anomalias.index, errors='coerce')

# 2. Identificar la columna de anomalías
col_maestra = 'anomaly_scores' if 'anomaly_scores' in df_anomalias.columns else 'anomaly_score'

# 3. Limpieza de strings en la columna de anomalías
df_anomalias[col_maestra] = df_anomalias[col_maestra].astype(str).str.replace(r'np\.float64', '', regex=True)
df_anomalias[col_maestra] = df_anomalias[col_maestra].str.replace(r'[\[\]\(\)]', '', regex=True)
df_anomalias[col_maestra] = pd.to_numeric(df_anomalias[col_maestra], errors='coerce')

# 4. ORDENAR TODO EL DATAFRAME
# Quitamos los vacíos en el score y ordenamos la tabla entera guiándonos por col_maestra
df_completo_ordenado = df_anomalias.dropna(subset=[col_maestra]).sort_values(by=col_maestra, ascending=False)

# 5. Exportar el resultado con todas las columnas
df_completo_ordenado.to_csv("Ranking_Completo_Con_Caracteristicas.csv", header=True)
print("✅ ¡Listo! Se creó 'Ranking_Completo_Con_Caracteristicas.csv'.")
