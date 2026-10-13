"""
app_dashboard_smieae.py

Dashboard SMIEAE (Streamlit): carga fisiológica vs. estrés y ansiedad.
Lee SMIEAE_Reporte_Completo.csv generado por 01_exploracion_estudiante1_eote.py.

Ejecutar:  streamlit run dashboard/app_dashboard_smieae.py --server.port 8506
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard SMIEAE", page_icon="🧠", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 SMIEAE: Carga Fisiológica vs. Estrés y Ansiedad</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D; font-size: 18px;'>Modelo EOTE: Entrenado exclusivamente en Días de Paz</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. CARGA Y LIMPIEZA DE DATOS
# ==========================================
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("SMIEAE_Reporte_Completo.csv", index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index()
        
        col_maestra = 'anomaly_scores' if 'anomaly_scores' in df.columns else 'anomaly_score'
        col_estres = 'stress' 
        col_ansiedad = 'anxiety'
        
        if col_maestra not in df.columns:
            st.error(f"🚨 No se encontró la columna '{col_maestra}'. Verifica el archivo.")
            st.stop()
            
        df[col_maestra] = df[col_maestra].astype(str).str.replace(r'np\.float64', '', regex=True)
        df[col_maestra] = df[col_maestra].str.replace(r'[\[\]\(\)]', '', regex=True)
        df[col_maestra] = pd.to_numeric(df[col_maestra], errors='coerce')
        df = df.dropna(subset=[col_maestra])
        
        # Mantenemos los NaN puros en las encuestas para los huecos vacíos
        if col_estres in df.columns:
            df[col_estres] = pd.to_numeric(df[col_estres], errors='coerce')
        else:
            df[col_estres] = np.nan
            
        if col_ansiedad in df.columns:
            df[col_ansiedad] = pd.to_numeric(df[col_ansiedad], errors='coerce')
        else:
            df[col_ansiedad] = np.nan
            
        # Aseguramos que exista la columna de Fase de Experimento
        if 'Fase_Experimento' not in df.columns:
            df['Fase_Experimento'] = 'Evaluación Global'
            
        return df, col_maestra, col_estres, col_ansiedad
    except FileNotFoundError:
        st.error("🚨 No se encontró el archivo 'SMIEAE_Reporte_Completo.csv'.")
        st.stop()

df, col_maestra, col_estres, col_ansiedad = cargar_datos()

# Calculamos el Umbral de Paz (Percentil 25) dinámicamente para graficar la línea horizontal
puntajes_diarios = df.groupby(df.index.date)[col_estres].first().dropna()
umbral_paz_personal = np.percentile(puntajes_diarios, 25) if not puntajes_diarios.empty else 0

# ==========================================
# 3. BARRA LATERAL (SLIDER GLOBAL Y FASE)
# ==========================================
st.sidebar.header("⚙️ Ajuste de Sensibilidad")

percentil_usuario = st.sidebar.slider("Percentil de Corte Global (%):", min_value=1, max_value=99, value=90, step=1)
umbral_EOTE = df[col_maestra].quantile(percentil_usuario / 100.0)

st.sidebar.success(f"""
**Regla de Clasificación EOTE:**
Corte: **{umbral_EOTE:.4f}**
✅ **< {umbral_EOTE:.4f}** = Normalidad
🚨 **>= {umbral_EOTE:.4f}** = Anomalía
""")

st.sidebar.markdown("---")
st.sidebar.info(f"""
📚 **Metodología de Entrenamiento**

El modelo EOTE aprendió la normalidad del alumno usando los **Días de Paz**.
Cualquier día con un estrés reportado **<= {umbral_paz_personal:.0f}** fue usado como línea base biológica.
""")

# ==========================================
# 4. MATEMÁTICA TEMPORAL Y CALENDARIO ACADÉMICO
# ==========================================
df['Es_Anomalia_Binario'] = np.where(df[col_maestra] >= umbral_EOTE, 1, 0)
df['Estado_Modelo'] = np.where(df['Es_Anomalia_Binario'] == 1, 'Evento Anómalo', 'Fisiología Normal')

# Ventana de acumulación a 60 minutos
df['Acumulado_60m'] = df['Es_Anomalia_Binario'].rolling('60min').sum()

meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 
             7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
df['Mes_Num'] = df.index.month
df['Mes_Nombre'] = df['Mes_Num'].map(meses_map)
df['Solo_Dia'] = df.index.date
df['Hora_Minuto'] = df.index.strftime('%H:%M')

picos_diarios = df.groupby(df.index.date)['Acumulado_60m'].max()
volumen_diario = df.groupby(df.index.date)['Es_Anomalia_Binario'].sum()

año_datos = df.index.year.value_counts().idxmax() if not df.empty else 2025
periodos_academicos = [
    {"nombre": "Semana 1", "inicio": f"{año_datos}-02-09", "fin": f"{año_datos}-02-15", "color": "#3498DB"},
    {"nombre": "Semana 2", "inicio": f"{año_datos}-02-16", "fin": f"{año_datos}-02-22", "color": "#2ECC71"},
    {"nombre": "Semana 3", "inicio": f"{año_datos}-02-23", "fin": f"{año_datos}-03-01", "color": "#F1C40F"},
    {"nombre": "Semana 4", "inicio": f"{año_datos}-03-02", "fin": f"{año_datos}-03-08", "color": "#E67E22"},
    {"nombre": "Semana 5", "inicio": f"{año_datos}-03-09", "fin": f"{año_datos}-03-15", "color": "#E74C3C"},
    {"nombre": "Semana Tec 1", "inicio": f"{año_datos}-03-16", "fin": f"{año_datos}-03-22", "color": "#9B59B6"},
    {"nombre": "Semana 1", "inicio": f"{año_datos}-03-23", "fin": f"{año_datos}-03-29", "color": "#3498DB"},
    {"nombre": "Semana 2", "inicio": f"{año_datos}-03-30", "fin": f"{año_datos}-04-05", "color": "#2ECC71"},
    {"nombre": "Semana 3", "inicio": f"{año_datos}-04-06", "fin": f"{año_datos}-04-12", "color": "#F1C40F"},
    {"nombre": "Vacaciones", "inicio": f"{año_datos}-04-13", "fin": f"{año_datos}-04-19", "color": "#BDC3C7"},
    {"nombre": "Semana 4", "inicio": f"{año_datos}-04-20", "fin": f"{año_datos}-04-26", "color": "#E67E22"},
    {"nombre": "Semana 5", "inicio": f"{año_datos}-04-27", "fin": f"{año_datos}-05-03", "color": "#E74C3C"},
    {"nombre": "Semana Tec 2", "inicio": f"{año_datos}-05-04", "fin": f"{año_datos}-05-10", "color": "#9B59B6"},
    {"nombre": "Semana 1", "inicio": f"{año_datos}-05-11", "fin": f"{año_datos}-05-17", "color": "#3498DB"},
    {"nombre": "Semana 2", "inicio": f"{año_datos}-05-18", "fin": f"{año_datos}-05-24", "color": "#2ECC71"},
    {"nombre": "Semana 3", "inicio": f"{año_datos}-05-25", "fin": f"{año_datos}-05-31", "color": "#F1C40F"},
    {"nombre": "Semana 4", "inicio": f"{año_datos}-06-01", "fin": f"{año_datos}-06-07", "color": "#E67E22"},
    {"nombre": "Semana 5", "inicio": f"{año_datos}-06-08", "fin": f"{año_datos}-06-14", "color": "#E74C3C"},
    {"nombre": "Semana Tec 3", "inicio": f"{año_datos}-06-15", "fin": f"{año_datos}-06-21", "color": "#9B59B6"},
]

# ==========================================
# 4.5. SISTEMA DE ALERTAS GLOBALES
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🚨 Criterio de Crisis Diaria")
percentil_alarma = st.sidebar.slider("Percentil Histórico de Alarma (%):", min_value=75, max_value=99, value=75, step=1)
umbral_alarma = picos_diarios.quantile(percentil_alarma / 100.0)

st.sidebar.error(f"**Regla de Disparo:** Se requieren **{umbral_alarma:.0f} bloques** continuos para marcar un día como Anómalo.")

dias_criticos = picos_diarios[picos_diarios >= umbral_alarma].sort_values(ascending=False)

# ==========================================
# 5. ESTRUCTURA DE PESTAÑAS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " 1. Resumen Global", 
    " 2. Desglose por Días", 
    " 3. Análisis de Anomalías",
    " 4. Días Normales vs Anómalos",
    " 5. Semáforo (Picos)",
    " 6. Estrés Crónico (Volumen)"
])

with tab1:
    st.subheader("Distribución Global de Eventos Fisiológicos")
    resumen_mensual = df.groupby(['Mes_Num', 'Mes_Nombre', 'Estado_Modelo']).size().reset_index(name='Cantidad')
    resumen_mensual = resumen_mensual.sort_values('Mes_Num')
    if not resumen_mensual.empty:
        fig_mes = px.bar(
            resumen_mensual, x='Mes_Nombre', y='Cantidad', color='Estado_Modelo', 
            barmode='group', color_discrete_map={'Fisiología Normal': '#2ca02c', 'Evento Anómalo': '#d62728'}
        )
        fig_mes.update_layout(plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
        st.plotly_chart(fig_mes, use_container_width=True)

with tab2:
    st.subheader("Validación: Anomalías Fisiológicas vs Reporte Subjetivo")
    st.markdown(f"Los días que caen **por debajo de la línea gris punteada (Umbral <= {umbral_paz_personal:.0f})** representan la normalidad biológica que el modelo usó para aprender.")
    
    meses_disponibles = df[['Mes_Num', 'Mes_Nombre']].drop_duplicates().sort_values('Mes_Num')['Mes_Nombre'].tolist()
    mes_seleccionado = st.selectbox("📆 Selecciona el mes a inspeccionar:", meses_disponibles)
    df_mes = df[df['Mes_Nombre'] == mes_seleccionado]
    
    resumen_contado = df_mes.groupby(['Solo_Dia', 'Estado_Modelo']).size().reset_index(name='Cantidad')
    pivot_diario = resumen_contado.pivot(index='Solo_Dia', columns='Estado_Modelo', values='Cantidad').fillna(0).reset_index()
    
    if 'Fisiología Normal' not in pivot_diario.columns: pivot_diario['Fisiología Normal'] = 0
    if 'Evento Anómalo' not in pivot_diario.columns: pivot_diario['Evento Anómalo'] = 0
    
    encuesta_por_dia = df_mes.groupby('Solo_Dia')[[col_estres, col_ansiedad]].first().reset_index()
    resumen_diario = pd.merge(pivot_diario, encuesta_por_dia, on='Solo_Dia', how='left')
    
    if not resumen_diario.empty:
        fig_diaria = go.Figure()
        
        # Barras de fisiología
        fig_diaria.add_trace(go.Bar(
            x=resumen_diario['Solo_Dia'], y=resumen_diario['Fisiología Normal'],
            name='Fisiología Normal (Paz)', marker_color='#2ca02c', yaxis='y1'
        ))
        
        fig_diaria.add_trace(go.Bar(
            x=resumen_diario['Solo_Dia'], y=resumen_diario['Evento Anómalo'],
            name='Eventos Anómalos', marker_color='#d62728', yaxis='y1'
        ))
        
        # Línea de Estrés
        fig_diaria.add_trace(go.Scatter(
            x=resumen_diario['Solo_Dia'], y=resumen_diario[col_estres],
            name='Estrés Reportado', mode='lines+markers', marker_color='#3498DB', 
            yaxis='y2', line=dict(width=3), connectgaps=False
        ))

        # Línea de Ansiedad
        fig_diaria.add_trace(go.Scatter(
            x=resumen_diario['Solo_Dia'], y=resumen_diario[col_ansiedad],
            name='Ansiedad Reportada', mode='lines+markers', marker_color='#9B59B6', 
            yaxis='y2', line=dict(width=3, dash='dot'), connectgaps=False
        ))

        # Línea Horizontal (Umbral de Paz)
        fig_diaria.add_trace(go.Scatter(
            x=[resumen_diario['Solo_Dia'].min(), resumen_diario['Solo_Dia'].max()],
            y=[umbral_paz_personal, umbral_paz_personal],
            mode='lines',
            name=f'Días de Entrenamiento (Paz <= {umbral_paz_personal:.0f})',
            line=dict(color='rgba(149, 165, 166, 0.8)', dash='dash', width=2),
            yaxis='y2', hoverinfo='skip'
        ))
        
        fecha_minima = df_mes['Solo_Dia'].min()
        fecha_maxima = df_mes['Solo_Dia'].max()
        
        # Calendario Académico (Semanas)
        for p in periodos_academicos:
            p_inicio = datetime.datetime.strptime(p["inicio"], "%Y-%m-%d").date()
            p_fin = datetime.datetime.strptime(p["fin"], "%Y-%m-%d").date()
            if p_inicio <= fecha_maxima and p_fin >= fecha_minima:
                fig_diaria.add_vrect(
                    x0=max(p_inicio, fecha_minima), x1=min(p_fin, fecha_maxima), 
                    fillcolor=p["color"], opacity=0.15, line_width=0, 
                    annotation_text=p["nombre"], annotation_position="top left", annotation_textangle=-90
                )
        
        fig_diaria.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", barmode='stack',
            xaxis=dict(title='Día', tickangle=-45, type='date', tickmode='linear', dtick=86400000),
            yaxis=dict(title='Minutos Registrados por el Reloj', side='left', showgrid=False),
            yaxis2=dict(title='Reporte Psicológico (0-100)', side='right', overlaying='y', range=[0, 100], showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_diaria, use_container_width=True)


with tab3:
    st.subheader("⚖️ Comparador de Días Anómalos")
    anomalias_por_dia = df.groupby('Solo_Dia')['Es_Anomalia_Binario'].sum()
    dias_con_anomalias = anomalias_por_dia[anomalias_por_dia > 0].index.tolist()
    
    if len(dias_con_anomalias) < 2:
        st.warning(f"Con el percentil actual ({percentil_usuario}%), no hay suficientes días con anomalías para comparar.")
    else:
        max_y_global = df['Acumulado_60m'].max() + 2
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("### 🚨 Día Anómalo A")
            dia_1 = st.selectbox("Selecciona el primer día:", sorted(dias_con_anomalias), index=0, key='d1')
            df_d1 = df[df['Solo_Dia'] == dia_1].copy().resample('5min').asfreq()
            
            e1_es = df_d1[col_estres].dropna().iloc[0] if not df_d1[col_estres].dropna().empty else np.nan
            e1_an = df_d1[col_ansiedad].dropna().iloc[0] if not df_d1[col_ansiedad].dropna().empty else np.nan
            st.metric(label="Estrés / Ansiedad", value=f"{'N/A' if pd.isna(e1_es) else f'{e1_es:.0f}'} / {'N/A' if pd.isna(e1_an) else f'{e1_an:.0f}'}")
            
            fig1 = px.area(df_d1, x=df_d1.index, y='Acumulado_60m', labels={'index': 'Hora', 'Acumulado_60m': 'Carga'})
            fig1.update_traces(line_color='#d62728', fillcolor='rgba(214, 39, 40, 0.3)', connectgaps=False)
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", yaxis_range=[0, max_y_global])
            st.plotly_chart(fig1, use_container_width=True)
            
        with colB:
            st.markdown("### 🚨 Día Anómalo B")
            dia_2 = st.selectbox("Selecciona el segundo día:", sorted(dias_con_anomalias), index=1, key='d2')
            df_d2 = df[df['Solo_Dia'] == dia_2].copy().resample('5min').asfreq()
            
            e2_es = df_d2[col_estres].dropna().iloc[0] if not df_d2[col_estres].dropna().empty else np.nan
            e2_an = df_d2[col_ansiedad].dropna().iloc[0] if not df_d2[col_ansiedad].dropna().empty else np.nan
            st.metric(label="Estrés / Ansiedad", value=f"{'N/A' if pd.isna(e2_es) else f'{e2_es:.0f}'} / {'N/A' if pd.isna(e2_an) else f'{e2_an:.0f}'}")
            
            fig2 = px.area(df_d2, x=df_d2.index, y='Acumulado_60m', labels={'index': 'Hora', 'Acumulado_60m': 'Carga'})
            fig2.update_traces(line_color='#d62728', fillcolor='rgba(214, 39, 40, 0.3)', connectgaps=False)
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", yaxis_range=[0, max_y_global])
            st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------
# Bloque para las tablas
# --------------------------------------------------------------------------
encuestas_es_dict = df.groupby(df.index.date)[col_estres].first()
encuestas_an_dict = df.groupby(df.index.date)[col_ansiedad].first()

def formatear_NaN(x):
    return 'Sin reporte' if pd.isna(x) else f'{x:.0f}'

def clasificar_uso_entrenamiento(x):
    if pd.isna(x): return "🔮 Evaluación"
    return "📚 Base de Paz" if x <= umbral_paz_personal else "🔮 Evaluación"

with tab4:
    st.subheader("📋 Clasificación Diaria: Anómalos vs. Normales")
    df_clasificacion = picos_diarios.reset_index()
    df_clasificacion.columns = ['Fecha', 'Pico Máximo (60m)']
    
    df_clasificacion['Estrés Reportado'] = df_clasificacion['Fecha'].map(encuestas_es_dict)
    df_clasificacion['Ansiedad Reportada'] = df_clasificacion['Fecha'].map(encuestas_an_dict)
    df_clasificacion['Rol del Modelo'] = df_clasificacion['Estrés Reportado'].apply(clasificar_uso_entrenamiento)

    df_clasificacion['Estado del Día'] = np.where(df_clasificacion['Pico Máximo (60m)'] >= umbral_alarma, '🚨 Día Anómalo', '✅ Día Normal')
    df_clasificacion = df_clasificacion.sort_values(by=['Estado del Día', 'Pico Máximo (60m)'], ascending=[True, False])

    def color_estado(val):
        return 'color: #D32F2F; font-weight: bold; background-color: #FADBD8;' if val == '🚨 Día Anómalo' else 'color: #1E8449; background-color: #D5F5E3;'

    st.dataframe(
        df_clasificacion.style.map(color_estado, subset=['Estado del Día']).format({
            'Pico Máximo (60m)': '{:.0f}',
            'Estrés Reportado': formatear_NaN,
            'Ansiedad Reportada': formatear_NaN
        }),
        use_container_width=True
    )

with tab5:
    st.subheader("🚦 Semáforo de Estrés Agudo (Picos)")
    df_semaforo = picos_diarios.reset_index()
    df_semaforo.columns = ['Fecha', 'Pico Máximo (60m)']
    
    df_semaforo['Estrés Reportado'] = df_semaforo['Fecha'].map(encuestas_es_dict)
    df_semaforo['Ansiedad Reportada'] = df_semaforo['Fecha'].map(encuestas_an_dict)

    condiciones = [
        (df_semaforo['Pico Máximo (60m)'] >= 7),
        (df_semaforo['Pico Máximo (60m)'] >= 4),
        (df_semaforo['Pico Máximo (60m)'] >= 0)
    ]
    df_semaforo['Nivel de Alerta'] = np.select(condiciones, ['🔴 Rojo (Crítico)', '🟡 Amarillo (Alerta)', '🟢 Verde (Normal)'], default='Desconocido')
    df_semaforo = df_semaforo.sort_values(by='Pico Máximo (60m)', ascending=False)

    def pintar_semaforo(val):
        if 'Rojo' in val: return 'background-color: #FADBD8; color: #C0392B; font-weight: bold;'
        elif 'Amarillo' in val: return 'background-color: #FCF3CF; color: #B7950B; font-weight: bold;'
        elif 'Verde' in val: return 'background-color: #D5F5E3; color: #1E8449; font-weight: bold;'
        return ''

    st.dataframe(
        df_semaforo.style.map(pintar_semaforo, subset=['Nivel de Alerta']).format({
            'Pico Máximo (60m)': '{:.0f}',
            'Estrés Reportado': formatear_NaN,
            'Ansiedad Reportada': formatear_NaN
        }),
        use_container_width=True
    )

with tab6:
    st.subheader("🔋 Estrés Crónico (Volumen de Anomalías)")
    umbral_vol_rojo = volumen_diario.quantile(0.75)
    umbral_vol_amarillo = volumen_diario.quantile(0.50)

    df_volumen = volumen_diario.reset_index()
    df_volumen.columns = ['Fecha', 'Total_Bloques_Dia']
    df_volumen['Estrés Reportado'] = df_volumen['Fecha'].map(encuestas_es_dict)
    df_volumen['Ansiedad Reportada'] = df_volumen['Fecha'].map(encuestas_an_dict)

    condiciones_vol = [
        (df_volumen['Total_Bloques_Dia'] >= umbral_vol_rojo),
        (df_volumen['Total_Bloques_Dia'] >= umbral_vol_amarillo),
        (df_volumen['Total_Bloques_Dia'] >= 0)
    ]
    df_volumen['Densidad de Estrés'] = np.select(condiciones_vol, [
        '🔴 Estrés Crónico (Desgaste Constante)', 
        '🟡 Estrés Moderado (Fricción Intermitente)', 
        '🟢 Paz General (Día Relajado)'
    ], default='Desconocido')

    df_volumen = df_volumen.sort_values(by='Total_Bloques_Dia', ascending=False)

    st.dataframe(
        df_volumen.style.map(pintar_semaforo, subset=['Densidad de Estrés']).format({
            'Total_Bloques_Dia': '{:.0f}',
            'Estrés Reportado': formatear_NaN,
            'Ansiedad Reportada': formatear_NaN
        }),
        use_container_width=True
    )
