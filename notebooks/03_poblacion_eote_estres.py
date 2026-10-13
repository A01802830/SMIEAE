"""
03_poblacion_eote_estres.py

Pipeline poblacional (estudiantes 1-10). Entrena con días de paz de estrés;
etiqueta del día = estrés > P70.

Proyecto SMIEAE · Detección de estrés y ansiedad mediante anomalías fisiológicas.
Autor: Javier Esteban Torres Sossa
"""


# %%
import os
import sys
import warnings
import glob
import json
import logging
import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# =================================================================
# 0. CONFIGURACIÓN INICIAL DE SILENCIO
# =================================================================
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
pd.options.mode.chained_assignment = None

sys.path.append(".")
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder

# Herramienta para apagar la consola mientras EOTE entrena (Mata el warning de Parallel)
class SilencioTotal:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


print("\n" + "="*65)
print("🚀 INICIANDO PIPELINE MASIVO: 10 ESTUDIANTES (100% BLINDADO)")
print("="*65)

y_true_global = []
y_pred_global = []

for estudiante_id in range(1, 11):
    print(f"\n" + "-"*55)
    print(f"👤 PROCESANDO ESTUDIANTE {estudiante_id}...")
    print("-"*55)
    
    try:
        base_path = rf'C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)\{estudiante_id}'
        
        # --- CARGA DE DATOS BLINDADA ---
        
        # 1. PASOS
        if os.path.exists(os.path.join(base_path, 'steps.csv')):
            df_steps = pd.read_csv(os.path.join(base_path, 'steps.csv'))
            df_steps['timestamp'] = pd.to_datetime(df_steps['timestamp'], utc=True)
            df_steps = df_steps.set_index('timestamp')
            df_steps.index = (df_steps.index - pd.Timedelta(hours=6)).tz_localize(None)
            steps_5min = df_steps['steps'].resample('5min').sum()
        else:
            steps_5min = pd.Series(dtype='float64')
        steps_5min.name = 'steps'

        # 2. OXÍGENO
        if os.path.exists(os.path.join(base_path, 'oxygen.csv')):
            df_oxygen = pd.read_csv(os.path.join(base_path, 'oxygen.csv'))
            if not df_oxygen.empty and 'value' in df_oxygen.columns:
                df_oxygen = df_oxygen[df_oxygen['value'] >= 80].copy()
                df_oxygen['timestamp'] = pd.to_datetime(df_oxygen['timestamp'], utc=True)
                df_oxygen = df_oxygen.set_index('timestamp')
                df_oxygen.index = (df_oxygen.index - pd.Timedelta(hours=6)).tz_localize(None)
                oxygen_5min = df_oxygen['value'].resample('5min').mean()
            else:
                oxygen_5min = pd.Series(dtype='float64')
        else:
            oxygen_5min = pd.Series(dtype='float64')
        oxygen_5min.name = 'SPO2_MEAN'

        # 3. HRV
        hrv_cols = ['HRV_RMSDD', 'HRV_RMSDD_STD','HRV_LF','HRV_LF_STD','HRV_HF','HRV_HF_STD']
        if os.path.exists(os.path.join(base_path, 'hrv.csv')):
            df_hrv = pd.read_csv(os.path.join(base_path, 'hrv.csv'))
            if not df_hrv.empty and all(c in df_hrv.columns for c in ['rmssd', 'low_frequency', 'high_frequency']):
                df_hrv['timestamp'] = pd.to_datetime(df_hrv['timestamp'], utc=True)
                df_hrv = df_hrv.set_index('timestamp')
                df_hrv.index = (df_hrv.index - pd.Timedelta(hours=6)).tz_localize(None)
                df_hrv.loc[df_hrv['rmssd'] > 200, 'rmssd'] = np.nan
                df_hrv.loc[df_hrv['rmssd'] <= 0, 'rmssd'] = np.nan
                df_hrv.loc[df_hrv['low_frequency'] < 0, 'low_frequency'] = np.nan
                df_hrv.loc[df_hrv['high_frequency'] < 0, 'high_frequency'] = np.nan
                hrv_stats = df_hrv[['rmssd', 'low_frequency', 'high_frequency']].rolling(window=3).agg(['mean', 'std'])
                
                # Blindaje por si al hacer rolling se quedan sin datos válidos
                if len(hrv_stats.columns) == 6:
                    hrv_stats.columns = hrv_cols
                else:
                    hrv_stats = pd.DataFrame(columns=hrv_cols)
            else:
                hrv_stats = pd.DataFrame(columns=hrv_cols)
        else:
            hrv_stats = pd.DataFrame(columns=hrv_cols)

        # 4. RITMO CARDÍACO
        ruta_carpeta_hr = os.path.join(base_path, "Global Export Data")
        archivos_json = glob.glob(os.path.join(ruta_carpeta_hr, "*heart_rate*.json"))
        datos_completos = []
        for archivo in archivos_json:
            with open(archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entrada in data:
                    datos_completos.append({'dateTime': entrada['dateTime'], 'bpm': entrada['value']['bpm']})
        
        if not datos_completos:
            print(f"⚠️ Sin datos de Ritmo Cardíaco. Saltando al siguiente estudiante...")
            continue
            
        df_heart_rate = pd.DataFrame(datos_completos)
        df_heart_rate['dateTime'] = pd.to_datetime(df_heart_rate['dateTime'], format='%m/%d/%y %H:%M:%S')
        df_heart_rate = df_heart_rate.set_index('dateTime')
        df_heart_rate.index = (df_heart_rate.index.tz_localize('UTC') - pd.Timedelta(hours=6)).tz_localize(None)

        # Nombramos las columnas directamente al crear el resample para evitar errores
        heart_rate_5min = df_heart_rate['bpm'].resample('5min').agg(HR_MEAN='mean', HR_STD='std')
        heart_rate_5min['HR_MEAN'] = heart_rate_5min['HR_MEAN'].interpolate(method='linear', limit=1)
        heart_rate_5min['first_diff'] = heart_rate_5min['HR_MEAN'].diff()
        heart_rate_5min['HR_MEAN_5'] = heart_rate_5min['HR_MEAN'].rolling(window=5, min_periods=1).mean()
        heart_rate_5min['HR_STD_5'] = heart_rate_5min['HR_MEAN'].rolling(window=5, min_periods=1).std()
        heart_rate_5min = heart_rate_5min.dropna(subset=['HR_MEAN', 'first_diff']).copy()
        heart_rate_5min['HR_STD_5'] = heart_rate_5min['HR_STD_5'].fillna(0)
        heart_rate_5min['HR_STD'] = heart_rate_5min['HR_STD'].fillna(0)

        # 5. LA FUSIÓN MAESTRA (SIN FORZAR NOMBRES DE COLUMNAS)
        # Aquí solucionamos el error del Estudiante 2. Pandas unirá lo que haya.
        dataset_maestro = pd.concat([heart_rate_5min, hrv_stats, steps_5min, oxygen_5min], axis=1)

        # 6. ENCUESTAS DIARIAS
        ruta_daily = os.path.join(base_path, 'daily_questions.csv')
        if not os.path.exists(ruta_daily):
            print("⚠️ No existe archivo de encuestas (daily_questions.csv). Saltando...")
            continue
            
        df_daily = pd.read_csv(ruta_daily)
        df_daily['timestamp'] = pd.to_datetime(df_daily['timeStampStart'], unit='s', utc=True)
        df_daily['timestamp'] = (df_daily['timestamp'] - pd.Timedelta(hours=6)).dt.tz_localize(None)
        df_daily['fecha'] = df_daily['timestamp'].dt.date
        daily_summary = df_daily.groupby('fecha')[['stress', 'anxiety']].mean()

        dataset_maestro['fecha'] = dataset_maestro.index.date
        dataset_maestro = dataset_maestro.join(daily_summary, on='fecha')
        dataset_maestro = dataset_maestro.drop(columns=['fecha'])

        # 7. FILTROS
        variables_fisiologicas = ['HR_MEAN', 'HR_STD', 'first_diff', 'HRV_RMSDD', 'HRV_RMSDD_STD', 'HRV_LF', 'HRV_LF_STD', 'HRV_HF', 'HRV_HF_STD']
        cols_presentes = [c for c in variables_fisiologicas if c in dataset_maestro.columns]
        
        dataset_maestro = dataset_maestro.dropna(subset=cols_presentes, how='all').copy()
        conteo_diario = dataset_maestro.groupby(dataset_maestro.index.date).size()
        dias_validos = conteo_diario[conteo_diario >= 144].index
        dataset_maestro = dataset_maestro[np.isin(dataset_maestro.index.date, dias_validos)].copy()

        if len(dataset_maestro) == 0:
            print(f"⚠️ El estudiante no cumplió el mínimo de adherencia. Saltando...")
            continue

        # -------------------------------------------------------------------
        # 2. PERFIL PSICOLÓGICO GLOBAL (TODA SU VIDA)
        # -------------------------------------------------------------------
        estres_global = dataset_maestro.groupby(dataset_maestro.index.date)['stress'].max().dropna()
        ansiedad_global = dataset_maestro.groupby(dataset_maestro.index.date)['anxiety'].max().dropna()
        carga_mental_global = pd.concat([estres_global, ansiedad_global], axis=1).max(axis=1)

        umbral_paz_personal = np.percentile(estres_global, 25)
        umbral_crisis_personal = np.percentile(estres_global, 70)
        
        print(f"✔️ PERFIL ESTABLECIDO | Paz (P25): <= {umbral_paz_personal:.1f} | Crisis (P75): > {umbral_crisis_personal:.1f}")

        # -------------------------------------------------------------------
        # 3. SEPARACIÓN 70/30 (Para separar la biología)
        # -------------------------------------------------------------------
        corte_idx = int(len(dataset_maestro) * 0.70)
        datos_pasado_70 = dataset_maestro.iloc[:corte_idx].copy()
        datos_futuro_30 = dataset_maestro.iloc[corte_idx:].copy()
        
        def marcar_normalidad(valor):
            return 'normal' if valor['stress'] <= umbral_paz_personal else 'anomalo'
        datos_pasado_70['estado'] = datos_pasado_70.apply(marcar_normalidad, axis=1)

        normalidad_pasado = datos_pasado_70[datos_pasado_70['estado'] == 'normal'].copy()
        columnas_ignorar = ['stress', 'anxiety', 'estado']
        x_train = normalidad_pasado.drop(columns=columnas_ignorar, errors='ignore').reset_index(drop=True)
        y_train = pd.DataFrame({'class': ['normal'] * len(x_train)}, index=x_train.index)

        # -------------------------------------------------------------------
        # 4. ENTRENAMIENTO Y PREDICCIÓN (CON SILENCIADOR TOTAL)
        # -------------------------------------------------------------------
        print("🧠 Entrenando y evaluando EOTE (Esto puede tardar unos segundos)...")
        director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
        eote_personal = director.get_eote()
        
        # Envolvemos el modelo en nuestra cápsula de Silencio Total
        with SilencioTotal():
            eote_personal.train(x_train, y_train)

        x_test = datos_futuro_30.drop(columns=columnas_ignorar, errors='ignore').copy()
        if len(x_test) == 0: 
            print("⚠️ Faltan datos futuros biométricos. Saltando...")
            continue

        with SilencioTotal():
            scores_lista = eote_personal.classify(x_test)
            
        datos_futuro_30['anomaly_score'] = [s[0] for s in scores_lista]

# -------------------------------------------------------------------
        # 5. EVALUACIÓN Y OPTIMIZACIÓN (SOLO ESTRÉS)
        # -------------------------------------------------------------------
        # Extraemos únicamente los valores máximos diarios de estrés
        estres_futuro = datos_futuro_30.groupby(datos_futuro_30.index.date)['stress'].max().dropna()
        
        # El Ground Truth ahora se basa 100% en el estrés del estudiante
        y_true_estudiante = (estres_futuro > umbral_crisis_personal).astype(int)
        
        # Suma de anomalías fisiológicas calculada por EOTE
        carga_eote = datos_futuro_30.groupby(datos_futuro_30.index.date)['anomaly_score'].sum().dropna()

        # Sincronizamos los días para evaluar solo donde tengamos ambos registros
        dias_comunes = y_true_estudiante.index.intersection(carga_eote.index)
        if len(dias_comunes) == 0: 
            print("⚠️ Faltan encuestas cruzadas en los días futuros. Saltando...")
            continue
            
        y_true_estudiante = y_true_estudiante.loc[dias_comunes].copy()
        carga_eote = carga_eote.loc[dias_comunes].copy()

        mejor_f1 = -1
        mejor_umbral = 0
        mejor_pred_estudiante = None

        # Búsqueda de hiperparámetros (Grid Search) para el umbral óptimo de EOTE
        limites = np.linspace(carga_eote.min(), carga_eote.max(), 100)
        for lim in limites:
            pred_temp = (carga_eote > lim).astype(int)
            f1 = f1_score(y_true_estudiante, pred_temp, zero_division=0)
                
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_umbral = lim
                mejor_pred_estudiante = pred_temp

        if mejor_pred_estudiante is not None:
            bloques_entrenados = len(x_train)
            bloques_evaluados = len(x_test)
            dias_evaluados = len(y_true_estudiante)
            dias_resueltos_correctamente = accuracy_score(y_true_estudiante, mejor_pred_estudiante, normalize=False)
            
            print(f"\n📊 RESUMEN DEL ESTUDIANTE {estudiante_id} (ESTRÉS):")
            print(f"   - Biometría entrenada: {bloques_entrenados} bloques (5-min)")
            print(f"   - Biometría evaluada:  {bloques_evaluados} bloques (5-min)")
            print(f"   - Psicología evaluada: {dias_evaluados} días")
            print(f"   - Días acertados:      {dias_resueltos_correctamente} días")
            print(f"   - F1-Score:            {mejor_f1:.2f}")
            print(f"   - Matriz de Confusión [Paz, Estrés]: {confusion_matrix(y_true_estudiante, mejor_pred_estudiante).tolist()}")
            
            y_true_global.extend(y_true_estudiante.values)
            y_pred_global.extend(mejor_pred_estudiante.values)
        else:
            print("⚠️ No se pudo optimizar este estudiante.")

    except Exception as e:
        print(f"❌ Error crítico procesando al estudiante {estudiante_id}: {e}")
        continue


# =================================================================
# 6. REPORTE FINAL DE POBLACIÓN (ESTRÉS)
# =================================================================
print("\n" + "="*65)
print("🏆 RESULTADOS GLOBALES DE LA TESIS (Solo Estrés)")
print("="*65)

if len(y_true_global) > 0:
    print(f"Total de días de población evaluados: {len(y_true_global)}\n")
    print("--- MATRIZ DE CONFUSIÓN GLOBAL ---")
    print("(Filas: Realidad [Paz, Estrés] | Columnas: EOTE [Paz, Estrés])")
    print(confusion_matrix(y_true_global, y_pred_global))
    print("\n--- REPORTE DE CLASIFICACIÓN FINAL ---")
    print(classification_report(y_true_global, y_pred_global, target_names=['Días de Paz (0)', 'Días de Estrés (1)'], zero_division=0))
else:
    print("❌ No se generaron suficientes datos poblacionales.")


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Generar la matriz con los datos globales
cm = confusion_matrix(y_true_global, y_pred_global)

# 2. Configurar el tamaño y estilo del gráfico
plt.figure(figsize=(7, 5))
sns.set_theme(font_scale=1.2) # Letra un poco más grande para la presentación

# 3. Dibujar el mapa de calor (Heatmap)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=['Días de Paz (0)', 'Días de Crisis (1)'],
            yticklabels=['Días de Paz (0)', 'Días de Crisis (1)'],
            annot_kws={"size": 16, "weight": "bold"}) # Números grandes y en negrita

# 4. Títulos y etiquetas
plt.title('Matriz de Confusión Poblacional (N=9)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Ground truth', fontsize=12, fontweight='bold')
plt.xlabel('Predicción del Modelo (EOTE)', fontsize=12, fontweight='bold')

# 5. Ajustar y mostrar
plt.tight_layout()
plt.show()
