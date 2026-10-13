"""
06_correlacion_patrones_datos_faltantes.py

Correlación de Spearman entre patrones de sensores faltantes
(BPM/RMSSD/SPO2) y estrés/ansiedad. Estudiantes 20-35.

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
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# =====================================================================
# 1. CONFIGURACIÓN DEL BUCLE
# =====================================================================
# CAMBIA ESTO por la ruta donde están las carpetas "1", "2", "3", etc.
ruta_base = r'C:\Users\esteb\PycharmProjects\Learn_Python\Trabajo\SSAQS dataset (2)'

# Lista de estudiantes a procesar (ejemplo: del 1 al 10)
lista_estudiantes = range(20, 36) 

# =====================================================================
# 2. FUNCIONES DE EXTRACCIÓN
# =====================================================================
def cargar_datos_estudiante(ruta_estudiante):
    """Carga y preprocesa los datos de un solo estudiante"""
    
    # --- OXÍGENO ---
    ruta_oxy = os.path.join(ruta_estudiante, 'oxygen.csv')
    if os.path.exists(ruta_oxy):
        df_oxygen = pd.read_csv(ruta_oxy)
        df_oxygen = df_oxygen[df_oxygen['value'] >= 80] 
        df_oxygen['timestamp'] = pd.to_datetime(df_oxygen['timestamp'], utc=True)
        df_oxygen = df_oxygen.set_index('timestamp')
        df_oxygen.index = (df_oxygen.index - pd.Timedelta(hours=6)).tz_localize(None)
    else:
        df_oxygen = pd.DataFrame(columns=['value']) # DataFrame vacío si no existe

    # --- HRV ---
    ruta_hrv = os.path.join(ruta_estudiante, 'hrv.csv')
    if os.path.exists(ruta_hrv):
        df_hrv = pd.read_csv(ruta_hrv)
        df_hrv['timestamp'] = pd.to_datetime(df_hrv['timestamp'], utc=True)
        df_hrv = df_hrv.set_index('timestamp')
        df_hrv.index = (df_hrv.index - pd.Timedelta(hours=6)).tz_localize(None)
        df_hrv.loc[df_hrv['rmssd'] > 200, 'rmssd'] = np.nan
        df_hrv.loc[df_hrv['rmssd'] <= 0, 'rmssd'] = np.nan
    else:
        df_hrv = pd.DataFrame(columns=['rmssd'])

    # --- RITMO CARDÍACO ---
    ruta_carpeta = os.path.join(ruta_estudiante, "Global Export Data")
    busqueda = os.path.join(ruta_carpeta, "*heart_rate*.json")
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
    if not df_heart_rate.empty:
        df_heart_rate['dateTime'] = pd.to_datetime(df_heart_rate['dateTime'], format='%m/%d/%y %H:%M:%S')
        df_heart_rate = df_heart_rate.set_index('dateTime')
        df_heart_rate.index = (df_heart_rate.index.tz_localize('UTC') - pd.Timedelta(hours=6)).tz_localize(None)

    # --- ENCUESTAS ---
    ruta_daily = os.path.join(ruta_estudiante, 'daily_questions.csv')
    if os.path.exists(ruta_daily):
        df_daily = pd.read_csv(ruta_daily)
        df_daily['timestamp'] = pd.to_datetime(df_daily['timeStampStart'], unit='s', utc=True)
        df_daily['timestamp'] = (df_daily['timestamp'] - pd.Timedelta(hours=6)).dt.tz_localize(None)
        df_daily['fecha'] = df_daily['timestamp'].dt.date
        daily_summary = df_daily.groupby('fecha')[['stress', 'anxiety']].mean()
    else:
        daily_summary = pd.DataFrame(columns=['stress', 'anxiety'])
        
    return df_heart_rate, df_hrv, df_oxygen, daily_summary


def extraer_patrones_diarios(df_hr, df_hrv, df_oxy, df_surveys):
    """Genera la tabla de patrones de falla diarios para un estudiante"""
    # 1. Resamplear a 5 minutos (usamos if para evitar errores si está vacío)
    hr_res = df_hr['bpm'].resample('5min').mean() if not df_hr.empty else pd.Series(dtype=float)
    hrv_res = df_hrv['rmssd'].resample('5min').mean() if not df_hrv.empty else pd.Series(dtype=float)
    oxy_res = df_oxy['value'].resample('5min').mean() if not df_oxy.empty else pd.Series(dtype=float)
    
    df_crudo = pd.concat([hr_res, hrv_res, oxy_res], axis=1)
    df_crudo.columns = ['BPM', 'RMSSD', 'SPO2']
    
    if df_crudo.empty:
        return pd.DataFrame()

    # 2. Identificar el patrón
    mask = df_crudo[['BPM', 'RMSSD', 'SPO2']].isna()
    df_crudo['patron'] = mask.apply(lambda x: f"BPM{int(x['BPM'])}_RMSSD{int(x['RMSSD'])}_SPO2{int(x['SPO2'])}", axis=1)
    
    # 3. Calcular porcentajes diarios
    df_crudo['fecha'] = df_crudo.index.date
    conteo_patrones = df_crudo.groupby(['fecha', 'patron']).size().unstack(fill_value=0)
    porcentajes = conteo_patrones.div(conteo_patrones.sum(axis=1), axis=0)
    
    # 4. Cruzar con encuestas
    resultados = porcentajes.join(df_surveys, how='inner')
    return resultados

# =====================================================================
# 3. BUCLE PRINCIPAL (ITERANDO SOBRE ESTUDIANTES)
# =====================================================================
resultados_globales = []

print("=== INICIANDO EXTRACCIÓN DE POBLACIÓN ===")
for estudiante_id in lista_estudiantes:
    ruta_estudiante = os.path.join(ruta_base, str(estudiante_id))
    
    if not os.path.exists(ruta_estudiante):
        print(f"⚠️ Saltando estudiante {estudiante_id}: Carpeta no encontrada.")
        continue
        
    print(f"Procesando estudiante {estudiante_id}...")
    try:
        # Cargar datos
        df_hr, df_hrv, df_oxy, daily_summary = cargar_datos_estudiante(ruta_estudiante)
        
        # Extraer patrones
        resultados_diarios = extraer_patrones_diarios(df_hr, df_hrv, df_oxy, daily_summary)
        
        if not resultados_diarios.empty:
            resultados_globales.append(resultados_diarios)
            
    except Exception as e:
        print(f"❌ Error con estudiante {estudiante_id}: {e}")

# =====================================================================
# 4. CORRELACIÓN Y VISUALIZACIÓN GLOBAL
# =====================================================================
if resultados_globales:
    # Concatenar todos los estudiantes (llenando con 0 si un estudiante no tuvo un patrón)
    df_global = pd.concat(resultados_globales, axis=0).fillna(0)
    
    print("\n=== CORRELACIONES GLOBALES POR PATRÓN (ANSIEDAD VS ESTRÉS) ===")
    lista_resumen = []
    
    # Obtener solo las columnas que son patrones (excluyendo stress y anxiety)
    columnas_patrones = [c for c in df_global.columns if c not in ['stress', 'anxiety']]
    
    for patron in columnas_patrones:
        # Solo calculamos si el patrón ocurrió al menos en 1 momento en toda la población
        if df_global[patron].sum() > 0: 
            corr_ans = df_global[patron].corr(df_global['anxiety'], method='spearman')
            corr_stress = df_global[patron].corr(df_global['stress'], method='spearman')
            
            print(f"Patrón {patron} (1=Falla, 0=Bien):")
            print(f"  -> Correlación con Ansiedad: {corr_ans:.3f}")
            print(f"  -> Correlación con Estrés:   {corr_stress:.3f}")
            
            lista_resumen.append({'Patron': patron, 'Anxiety': corr_ans, 'Stress': corr_stress})
    
    # Graficar
    resumen_df = pd.DataFrame(lista_resumen).set_index('Patron')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(resumen_df, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.title(f"Correlación de Patrones de Falla (Población: {len(lista_estudiantes)} estudiantes)")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo se pudieron extraer datos de los estudiantes proporcionados.")
