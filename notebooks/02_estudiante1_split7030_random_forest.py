"""
02_estudiante1_split7030_random_forest.py

Estudiante 1: split cronológico 70/30 por días, carga emocional unificada,
EOTE + Random Forest sobre pilares diarios (Normalidad vs Crisis).

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# 1. Agrupación base a 5 minutos
df_heart_rate = pd.DataFrame(datos_completos)
df_heart_rate['dateTime'] = pd.to_datetime(df_heart_rate['dateTime'], format='%m/%d/%y %H:%M:%S')
df_heart_rate = df_heart_rate.set_index('dateTime')
df_heart_rate.index = (df_heart_rate.index.tz_localize('UTC') - pd.Timedelta(hours=6)).tz_localize(None)

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
# =================================================================
# FASE 1: MODELO BASE (EOTE)
# =================================================================

# 1. DIVISION CRONOLOGICA 70/30 (DIAS COMPLETOS INTACTOS)
dias_unicos = np.unique(dataset_maestro.index.date)
dias_unicos.sort()

corte_idx_dias = int(len(dias_unicos) * 0.70)
fecha_corte = dias_unicos[corte_idx_dias]

print("=================================================================")
print("FASE 1: ENTRENAMIENTO DEL EXTRACTOR DE CARACTERISTICAS (EOTE)")
print("=================================================================")
print(f"Total de dias validos del estudiante: {len(dias_unicos)}")
print(f"El 70% del semestre abarca hasta antes del dia: {fecha_corte}")

datos_pasado_70 = dataset_maestro[dataset_maestro.index.date < fecha_corte].copy()
datos_futuro_30 = dataset_maestro[dataset_maestro.index.date >= fecha_corte].copy()


# %%
# ===================================================================
# INTEGRACIÓN SEGURA: CARGA EMOCIONAL UNIFICADA
# ===================================================================

# 1. Definir los conjuntos (usando tu lógica de 70/30)
# Ya tenemos 'datos_pasado_70' y 'datos_futuro_30' definidos en tu flujo
emociones_pasado = datos_pasado_70.groupby(datos_pasado_70.index.date)[['stress', 'anxiety']].max().dropna()
emociones_futuro = datos_futuro_30.groupby(datos_futuro_30.index.date)[['stress', 'anxiety']].max().dropna()

# 2. Normalización (Entrenamiento ÚNICO con el pasado)
scaler = MinMaxScaler()
scaler.fit(emociones_pasado) # Aquí el modelo NO ve el futuro

# Transformamos ambos conjuntos
norm_pasado = pd.DataFrame(scaler.transform(emociones_pasado), columns=['s', 'a'], index=emociones_pasado.index)
norm_futuro = pd.DataFrame(scaler.transform(emociones_futuro), columns=['s', 'a'], index=emociones_futuro.index)

# 3. Crear la Carga Emocional (Indice compuesto)
cargas_diarias = pd.concat([norm_pasado.mean(axis=1), norm_futuro.mean(axis=1)])
cargas_diarias.name = 'carga_emocional_unificada'

# 4. Calcular Umbrales SOLO con el pasado (para evitar sesgos)
p25_carga = norm_pasado.mean(axis=1).quantile(0.25)
p70_carga = norm_pasado.mean(axis=1).quantile(0.70)

# 5. Integrar al Dataset Maestro
# Mapeamos usando la fecha normalizada del dataset maestro
dataset_maestro['fecha_aux'] = dataset_maestro.index.normalize()
dataset_maestro['carga_emocional_unificada'] = dataset_maestro['fecha_aux'].map(cargas_diarias)

# Crear la etiqueta (estado)
dataset_maestro['estado'] = pd.cut(
    dataset_maestro['carga_emocional_unificada'],
    bins=[-np.inf, p25_carga, p70_carga, np.inf],
    labels=['Paz', 'Medio', 'Crisis']
)

dataset_maestro = dataset_maestro.drop(columns=['fecha_aux'])

print(f"Integración completa. Umbrales basados en el 70% del pasado:")
print(f"Paz (P25): <= {p25_carga:.4f} | Crisis (P70): > {p70_carga:.4f}")


# %%
df_entrenamiento =dataset_maestro[dataset_maestro['estado'] == 'Paz']
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
# =================================================================
# FASE 1: MODELO BASE (EOTE)
# =================================================================

# 1. DIVISION CRONOLOGICA 70/30 (DIAS COMPLETOS INTACTOS)
dias_unicos = np.unique(dataset_maestro.index.date)
dias_unicos.sort()

corte_idx_dias = int(len(dias_unicos) * 0.70)
fecha_corte = dias_unicos[corte_idx_dias]

print("=================================================================")
print("FASE 1: ENTRENAMIENTO DEL EXTRACTOR DE CARACTERISTICAS (EOTE)")
print("=================================================================")
print(f"Total de dias validos del estudiante: {len(dias_unicos)}")
print(f"El 70% del semestre abarca hasta antes del dia: {fecha_corte}")

datos_pasado_70 = dataset_maestro[dataset_maestro.index.date < fecha_corte].copy()
datos_futuro_30 = dataset_maestro[dataset_maestro.index.date >= fecha_corte].copy()
# 2. EXTRAER "NORMALIDAD PURA" SOLO DEL 70% DEL PASADO
normalidad_pasado = datos_pasado_70[datos_pasado_70['estado'] == 'Paz'].copy()
columnas_ignorar = ['stress', 'anxiety', 'estado','carga_emocional_unificada']

x_train_70 = normalidad_pasado.drop(columns=columnas_ignorar, errors='ignore').reset_index(drop=True)
y_train_70 = pd.DataFrame({'class': ['normal'] * len(x_train_70)}, index=x_train_70.index)

# 3. ENTRENAMIENTO DEL MODELO EOTE PREDICTIVO
print(f"\nEntrenando EOTE Predictivo con {len(x_train_70)} bloques puros del PASADO...")

sys.path.append(".")
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder

director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
eote_predictivo = director.get_eote()
eote_predictivo.train(x_train_70, y_train_70)

# 4. PREDICCION SOBRE PASADO Y FUTURO (PARA ALIMENTAR A XGBOOST)
print("\nGenerando Puntajes de Anomalia (Anomaly Scores) con EOTE...")
x_past_all = datos_pasado_70.drop(columns=columnas_ignorar, errors='ignore')
x_test_30 = datos_futuro_30.drop(columns=columnas_ignorar, errors='ignore')

# Calificamos el pasado (para que XGBoost entrene) y el futuro (para evaluar)
datos_pasado_70['anomaly_score'] = [score[0] for score in eote_predictivo.classify(x_past_all)]
datos_futuro_30['anomaly_score'] = [score[0] for score in eote_predictivo.classify(x_test_30)]


# %%
import pandas as pd
import numpy as np

def construir_arquitectura_definitiva(df_bloques, umbral_anomalia):
    """
    Construye la matriz de características diarias evaluando 
    la suma total, la presencia de estrés y múltiples dimensiones de paz.
    """
    df_temp = df_bloques.copy().sort_index()
    
    def extraer_perfil_diario(datos_dia):
        # 1. Separar lo normal de lo anómalo según el umbral
        es_paz = datos_dia['anomaly_score'] <= umbral_anomalia
        
        bloques_anomalos = datos_dia[~es_paz]
        bloques_paz = datos_dia[es_paz]
        
        # --- PILARES GLOBALES ---
    
        
        # --- PILARES DE ESTRÉS / CRISIS ---
        carga_total = bloques_anomalos['anomaly_score'].sum() if not bloques_anomalos.empty else 0
        severidad_promedio = bloques_anomalos['anomaly_score'].mean() if not bloques_anomalos.empty else 0
        
        trenes_30m = datos_dia['anomaly_score'].rolling(window=6, min_periods=1).sum()
        tren_critico = trenes_30m.max()
        
        # --- PILARES DE PAZ / RECUPERACIÓN ---
        total_bloques_registrados = len(datos_dia)
        indice_paz = (len(bloques_paz) / total_bloques_registrados) * 100 if total_bloques_registrados > 0 else 0
 
        # NUEVO: Racha Máxima de Paz Continua (en minutos)
        # Agrupa los bloques consecutivos que cumplen la condición de paz y suma cuántos son
        rachas = es_paz.groupby((es_paz != es_paz.shift()).cumsum()).sum()
        racha_maxima_minutos = rachas.max() * 5 if not rachas.empty else 0
        
        return pd.Series({
            
            'Carga_Total_Estres': carga_total,
            'Severidad_Promedio': severidad_promedio,
            'Tren_Critico_30m': tren_critico,
            'Indice_Paz_Porcentual': indice_paz,
        
            'Racha_Paz_Maxima_Minutos': racha_maxima_minutos
        })
    
    # Agrupamos por día y extraemos el perfil
    arquitectura_diaria = df_temp.groupby(df_temp.index.date).apply(extraer_perfil_diario).fillna(0)
    
    return arquitectura_diaria

# =================================================================
# APLICACIÓN AL PIPELINE
# =================================================================
# Calculamos el umbral usando el PASADO (P90)
umbral_EOTE = datos_pasado_70['anomaly_score'].quantile(0.90)
print(f"Umbral para considerar un bloque de 5 min como 'anómalo': {umbral_EOTE:.2f}")

# Construimos la arquitectura
X_train_arq = construir_arquitectura_definitiva(datos_pasado_70, umbral_EOTE)
X_test_arq = construir_arquitectura_definitiva(datos_futuro_30, umbral_EOTE)

print("\nVista previa de la Arquitectura (Primeros 5 días del pasado):")
print(X_train_arq.head())


# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=================================================================")
print("MODELO BINARIO FINAL: NORMALIDAD VS CRISIS (Usando los 8 Pilares)")
print("=================================================================")

# =================================================================
# 1. ALINEACIÓN Y MAPEO BINARIO (GROUND TRUTH)
# =================================================================
# Extraemos el 'estado' original calculado en el dataset_maestro
y_train_str = datos_pasado_70.groupby(datos_pasado_70.index.date)['estado'].first().dropna()
y_test_str = datos_futuro_30.groupby(datos_futuro_30.index.date)['estado'].first().dropna()

# Alinear índices rigurosamente para que X e Y coincidan en fechas
idx_train = X_train_arq.index.intersection(y_train_str.index)
X_train_final = X_train_arq.loc[idx_train]
y_train_str = y_train_str.loc[idx_train]

idx_test = X_test_arq.index.intersection(y_test_str.index)
X_test_final = X_test_arq.loc[idx_test]
y_test_str = y_test_str.loc[idx_test]

# Transformamos a Binario (Cubrimos todos los posibles nombres que hayas usado)
mapeo_binario = {
    'Paz': 'Normalidad', 
    'Medio': 'Normalidad', 
    'Crisis': 'Crisis',
    'normal': 'Normalidad',
    'anomalo': 'Crisis'
}
y_train_bin = y_train_str.map(mapeo_binario)
y_test_bin = y_test_str.map(mapeo_binario)

print(f"Distribución en Entrenamiento (70%): \n{y_train_bin.value_counts()}\n")

# =================================================================
# 2. ENTRENAMIENTO DEL RANDOM FOREST
# =================================================================
# Usamos class_weight='balanced' para que la IA no ignore la Crisis
modelo_binario = RandomForestClassifier(
    n_estimators=200,
    max_depth=4, 
    class_weight='balanced', 
    random_state=42
)

modelo_binario.fit(X_train_final, y_train_bin)

# =================================================================
# 3. PREDICCIÓN Y REPORTE EN EL FUTURO (30%)
# =================================================================
predicciones_bin = modelo_binario.predict(X_test_final)

print("--- REPORTE DE CLASIFICACIÓN (DATOS INÉDITOS 30%) ---")
print(classification_report(y_test_bin, predicciones_bin, labels=['Normalidad', 'Crisis']))

# =================================================================
# 4. MATRIZ DE CONFUSIÓN VISUAL
# =================================================================
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test_bin, predicciones_bin, labels=['Normalidad', 'Crisis'])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normalidad (Pred)', 'Crisis (Pred)'],
            yticklabels=['Normalidad (Real)', 'Crisis (Real)'])

plt.title('Capacidad del Modelo: Normalidad vs. Crisis', fontsize=13, fontweight='bold')
plt.ylabel('Realidad (Encuestas)', fontweight='bold')
plt.xlabel('Predicción (IA + Sensores)', fontweight='bold')
plt.tight_layout()
plt.show()

# =================================================================
# 5. EXPLICABILIDAD (PESO DE LOS 8 PILARES)
# =================================================================
importancias_bin = pd.Series(modelo_binario.feature_importances_, index=X_train_final.columns)
importancias_bin = importancias_bin.sort_values(ascending=True)

plt.figure(figsize=(9, 5))
# Usamos un color distintivo para la importancia
importancias_bin.plot(kind='barh', color='#8e44ad', edgecolor='black')
plt.title('¿Qué variable es más importante para predecir la Crisis?', fontsize=13, fontweight='bold')
plt.xlabel('Peso de Importancia (0 a 1)')
plt.ylabel('Características (Pilares)')
plt.tight_layout()
plt.show()
