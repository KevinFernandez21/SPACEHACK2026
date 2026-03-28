"""
🌿 Sistema de Monitoreo de Manglares — Gran Guayaquil
Dashboard + Chatbot · Datos basados en clasificación GEE Sentinel-1/2 y Random Forest
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import json, re
import random

# Nota: Asegúrate de configurar tu API KEY de Anthropic en tus variables de entorno
# o st.secrets para que el chatbot funcione, o sustituye con otra librería.
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    from anthropic import Anthropic
    anthropic_client = Anthropic()
    has_anthropic = True
except Exception:
    has_anthropic = False

# ════════════════════════════════════════════════════════════
# DATOS EXTRAÍDOS DEL NOTEBOOK GEE (K3)
# ════════════════════════════════════════════════════════════

np.random.seed(42)

# ── Áreas de manglar calculadas (Hectáreas) ────
MANGLARES_HA = {
    2018: 72952.2,  # SERVIR
    2020: 71459.9,  # SERVIR
    2022: 71199.0,  # SERVIR (Referencia)
    2023: 76407.6,  # Clasificación RF
    2024: 71868.0,  # Clasificación RF
    2025: 54709.5,  # Clasificación RF
}

# ── Datos de cambio de manglares 2022-2025 ─────────────────
CAMBIO_MANGLAR = {
    "persistente_ha": 51051.6,
    "ganancia_ha":    4736.0,
    "perdida_ha":     19471.9,
}

# ── Importancia de variables del modelo Random Forest ───────
BAND_IMPORTANCE = {
    "VH": 1098.73, "VV_VH_ratio": 991.58, "VV": 968.58, "B12": 952.42,
    "MVI": 943.46, "MMRI": 936.47, "NDBI": 921.67, "B11": 918.97,
    "B5": 917.89, "B3": 915.46, "MNDWI": 908.65, "slope": 902.97,
    "NDRE": 890.39, "B2": 885.08, "NDMI": 877.74, "NDWI": 855.12,
    "B8A": 854.51, "B4": 853.96, "B8": 846.18, "B7": 820.45,
    "elevation": 820.16, "B6": 818.69, "NDVI": 812.75, "SAVI": 793.14, "EVI": 780.69
}

# ── Generación de Datos Sintéticos Espaciales para el Mapa ──
def _generar_puntos_mapa():
    """Genera puntos sintéticos alrededor del Golfo de Guayaquil para visualización de cambios."""
    puntos = []
    # Centro aproximado del Golfo / Reserva Ecológica Manglares Churute
    base_lat, base_lon = -2.4, -79.8

    # Pesos basados en los resultados reales (más persistencia, luego pérdida, luego ganancia)
    estados = ["Persistente", "Pérdida", "Ganancia"]
    probs = [0.65, 0.25, 0.10]
    colores = {
        "Persistente": [0, 204, 68, 180],   # Verde
        "Pérdida": [255, 51, 51, 200],      # Rojo
        "Ganancia": [0, 102, 255, 180]      # Azul
    }

    for _ in range(800):
        # Distribución dispersa a lo largo del estuario
        lat = base_lat + np.random.normal(0, 0.15)
        lon = base_lon + np.random.normal(0, 0.15)
        estado = np.random.choice(estados, p=probs)

        puntos.append({
            "lat": lat,
            "lon": lon,
            "estado": estado,
            "color": colores[estado],
            "radio": random.randint(300, 800)
        })
    return pd.DataFrame(puntos)

@st.cache_data
def cargar_mapa():
    return _generar_puntos_mapa()

# ════════════════════════════════════════════════════════════
# CONTEXTO PARA EL CHATBOT
# ════════════════════════════════════════════════════════════

DATOS_CONTEXTO = f"""
SISTEMA DE MONITOREO DE MANGLARES — GRAN GUAYAQUIL, ECUADOR
Datos extraídos de plataforma de Google Earth Engine (Clasificación de 2018 a 2025).

ÁREA DE MANGLARES POR AÑO (hectáreas):
{json.dumps(MANGLARES_HA, indent=2)}

CAMBIOS DETECTADOS ENTRE 2022 Y 2025:
- Manglar Persistente: {CAMBIO_MANGLAR['persistente_ha']:,.1f} ha
- Ganancia de Manglar: {CAMBIO_MANGLAR['ganancia_ha']:,.1f} ha
- Pérdida de Manglar (Deforestación): {CAMBIO_MANGLAR['perdida_ha']:,.1f} ha

MÉTRICAS DEL MODELO (Random Forest, 300 árboles):
- Accuracy Global: 79.08%
- Índice Kappa: 0.7384
- Variables de mayor importancia: Radar SAR (VH, Ratio VV/VH, VV), Banda 12 (SWIR), e Índices MVI y MMRI.

CONTEXTO ECOLÓGICO Y AMENAZAS:
- Ecuador alberga extensiones críticas de manglar en el Golfo de Guayaquil (Churute, canales).
- Amenazas principales: Expansión de camaroneras, tala ilegal, contaminación y crecimiento urbano.
- Importancia: Los manglares son barreras naturales contra inundaciones, capturan "carbono azul" y son criaderos vitales para la pesca artesanal (concha, cangrejo).
"""

SYSTEM_PROMPT = f"""Eres un experto en ecología costera, conservación de manglares y teledetección satelital enfocado en el Golfo de Guayaquil, Ecuador.
Tu rol es asesorar a organizaciones ambientales, gobierno local y ciudadanos sobre el estado de los ecosistemas de manglar utilizando los datos proporcionados.

DATOS EXCLUSIVOS DEL SISTEMA:
{DATOS_CONTEXTO}

REGLAS DE RESPUESTA:
- Responde siempre en español con tono profesional, pero accesible.
- Usa formato de números claro: "72,952 hectáreas".
- Cuando te pregunten por motivos de pérdida, asocia las métricas a causas reales en Ecuador (piscinas camaroneras, urbanización, agricultura).
- Si te preguntan sobre el satélite o modelo, explica brevemente que se usó Sentinel-1 (radar) y Sentinel-2 (óptico) con un Random Forest de precisión 79%.
- Máximo 3 o 4 párrafos cortos por respuesta. Usa viñetas si listas datos.
"""

# ════════════════════════════════════════════════════════════
# STREAMLIT CONFIG + ESTILOS
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monitoreo Manglares · Guayaquil",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, html, body { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,.mono { font-family: 'IBM Plex Mono', monospace !important; }

/* Fondo principal oscuro */
.stApp { background: #0d1117; color: #c9d1d9; }
.main .block-container { padding: 1.5rem 2rem; max-width: 1440px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    padding: 10px 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #3fb950 !important;
    border-bottom: 2px solid #3fb950 !important;
    background: transparent !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    border-left: 4px solid #3fb950;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.7rem !important;
    color: #f0f6fc !important;
}

/* Botones */
.stButton button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em;
    transition: all 0.15s;
}
.stButton button:hover {
    background: #3fb950 !important;
    color: #0d1117 !important;
    border-color: #3fb950 !important;
}

/* Chat */
.msg-user {
    background: #238636;
    color: #fff;
    border-radius: 12px 12px 3px 12px;
    padding: 10px 16px;
    margin: 8px 0 8px 15%;
    font-size: 0.9rem;
    line-height: 1.5;
}
.msg-bot {
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-left: 3px solid #3fb950;
    border-radius: 3px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 15% 8px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.msg-bot b, .msg-bot strong { color: #56d364; }

/* Input chat */
.stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #f0f6fc !important;
    border-radius: 6px !important;
}
.stTextInput input:focus { border-color: #3fb950 !important; }

/* Header */
.page-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 4px;
}
.badge {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3fb950;
    padding: 2px 8px;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SESSION STATE & CHAT ENGINE
# ════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_mock(user_msg: str) -> str:
    """Mock responder fallback if API key is not set."""
    user_msg = user_msg.lower()
    if "2025" in user_msg or "actual" in user_msg or "cuanto" in user_msg:
        return "Según nuestros últimos datos satelitales (2025), el área total de manglares en el Gran Guayaquil se estima en **54,709.5 hectáreas**. Ha habido una reducción significativa en comparación con las 71,199 hectáreas registradas en 2022."
    elif "perdida" in user_msg or "pérdida" in user_msg or "deforest" in user_msg:
        return "Entre 2022 y 2025, el modelo detectó una **pérdida de 19,471.9 hectáreas** frente a una ganancia de solo 4,736.0 hectáreas. Históricamente, en el Golfo de Guayaquil, los principales impulsores de esta pérdida son la expansión de infraestructuras acuícolas (piscinas camaroneras) y el avance urbano."
    elif "modelo" in user_msg or "precision" in user_msg or "precisión" in user_msg:
        return "Utilizamos un algoritmo de **Random Forest (300 árboles)** procesando imágenes Sentinel-1 (Radar) y Sentinel-2 (Óptico). Alcanzamos una **precisión global del 79.08%** y un índice Kappa de 0.7384. Las bandas de radar (VH, VV) fueron las más útiles para penetrar la nubosidad y diferenciar la estructura del manglar."
    else:
        return "El ecosistema de manglares en Guayaquil es vital. Actúa como sumidero de carbono azul y protege las costas. Nuestras métricas muestran un balance preocupante: la deforestación supera ampliamente la regeneración natural. ¿Hay algún dato específico (años, modelo, área) sobre el que quieras profundizar?"

def procesar_mensaje(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Analizando datos..."):
        if has_anthropic:
            try:
                msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                r = anthropic_client.messages.create(
                    model="claude-3-haiku-20240307", # Haiku para rapidez, puedes cambiar a sonnet
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=msgs,
                )
                bot_resp = r.content[0].text
            except Exception as e:
                bot_resp = f"Error en API (usando mock de respaldo). Fallo: {str(e)}\n\n{chat_mock(user_input)}"
        else:
            bot_resp = chat_mock(user_input)

    st.session_state.messages.append({"role": "assistant", "content": bot_resp})
    st.rerun()

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
  <h1 style="color:#f0f6fc;font-size:1.5rem;margin:0">🌿 CONSERVACIÓN DE MANGLARES</h1>
  <span class="badge">GRAN GUAYAQUIL</span>
  <span class="badge">S1 SAR / S2 ÓPTICO / RANDOM FOREST</span>
</div>
<p style="color:#8b949e;font-size:0.82rem;margin:0 0 20px 0">
Monitoreo de deforestación y regeneración (2018-2025) · Apoyo a decisiones ambientales
</p>
""", unsafe_allow_html=True)

# Tabs
tab_dash, tab_map, tab_model, tab_chat = st.tabs([
    "📊  COBERTURA Y CAMBIOS", "🗺️  MAPA ESPACIAL", "⚙️  MÉTRICAS DEL MODELO", "💬  ASISTENTE ECOLÓGICO"
])

# ────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ────────────────────────────────────────────────────────────
with tab_dash:
    # Métricas superiores
    m1, m2, m3, m4 = st.columns(4)
    perdida_historica = MANGLARES_HA[2018] - MANGLARES_HA[2025]
    m1.metric("Área Actual (2025)", f"{MANGLARES_HA[2025]:,.1f} ha")
    m2.metric("Pérdida Neta (vs 2018)", f"{perdida_historica:,.1f} ha", f"-{(perdida_historica/MANGLARES_HA[2018]*100):.1f}%", delta_color="inverse")
    m3.metric("Deforestación (22-25)", f"{CAMBIO_MANGLAR['perdida_ha']:,.1f} ha")
    m4.metric("Regeneración (22-25)", f"{CAMBIO_MANGLAR['ganancia_ha']:,.1f} ha")

    g1, g2 = st.columns([3, 2])

    with g1:
        # Gráfico de Serie de Tiempo
        df_mg = pd.DataFrame([{"Año": k, "Área (ha)": v} for k, v in MANGLARES_HA.items()])
        fig_line = go.Figure()

        # Conectar los puntos
        fig_line.add_trace(go.Scatter(
            x=df_mg["Año"], y=df_mg["Área (ha)"],
            mode="lines+markers",
            line=dict(color="#3fb950", width=3),
            marker=dict(size=8, color="#3fb950"),
            fill="tozeroy",
            fillcolor="rgba(63, 185, 80, 0.1)",
            name="Superficie Manglar"
        ))

        fig_line.update_layout(
            title="Evolución de Cobertura de Manglar (2018 - 2025)",
            title_font=dict(family="IBM Plex Mono", size=13, color="#8b949e"),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(family="IBM Plex Sans", color="#8b949e"),
            xaxis=dict(gridcolor="#21262d", tickmode="array", tickvals=list(MANGLARES_HA.keys())),
            yaxis=dict(gridcolor="#21262d", title="Hectáreas"),
            margin=dict(t=40, b=40, l=40, r=20), height=320,
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with g2:
        # Gráfico Donut de Cambios Recientes
        fig_donut = go.Figure(go.Pie(
            labels=["Persistente", "Pérdida", "Ganancia"],
            values=[CAMBIO_MANGLAR["persistente_ha"], CAMBIO_MANGLAR["perdida_ha"], CAMBIO_MANGLAR["ganancia_ha"]],
            hole=0.6,
            marker=dict(colors=["#00cc44", "#ff3333", "#0066ff"]),
            textinfo="percent+label",
            textfont=dict(family="IBM Plex Mono", size=11, color="white"),
        ))
        fig_donut.update_layout(
            title="Dinámica de Cambio (2022 - 2025)",
            title_font=dict(family="IBM Plex Mono", size=13, color="#8b949e"),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            showlegend=False,
            margin=dict(t=40, b=20, l=10, r=10), height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 2: MAPA ESPACIAL
# ────────────────────────────────────────────────────────────
with tab_map:
    st.markdown('<p style="font-family:\'IBM Plex Mono\'; font-size: 0.8rem; color: #8b949e;">VISUALIZACIÓN DE CAMBIOS EN GOLFO DE GUAYAQUIL (DATOS SINTÉTICOS BASADOS EN GEE)</p>', unsafe_allow_html=True)

    df_map = cargar_mapa()

    col_m, col_l = st.columns([4, 1])

    with col_l:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;">
          <p style="font-family:'IBM Plex Mono';font-size:0.75rem;color:#8b949e;">LEYENDA</p>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <div style="width:12px;height:12px;background:#00cc44;border-radius:50%"></div>
            <span style="font-size:0.85rem;color:#c9d1d9">Persistente</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <div style="width:12px;height:12px;background:#ff3333;border-radius:50%"></div>
            <span style="font-size:0.85rem;color:#c9d1d9">Pérdida</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <div style="width:12px;height:12px;background:#0066ff;border-radius:50%"></div>
            <span style="font-size:0.85rem;color:#c9d1d9">Ganancia</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radio",
            pickable=True,
        )
        view = pdk.ViewState(
            latitude=-2.4, longitude=-79.8,
            zoom=8.5, pitch=35,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"html": "<b>Estado:</b> {estado}", "style": {"backgroundColor":"#161b22","color":"white"}},
            map_style="mapbox://styles/mapbox/dark-v10",
        ))

# ────────────────────────────────────────────────────────────
# TAB 3: MODELO GEE
# ────────────────────────────────────────────────────────────
with tab_model:
    st.markdown('<p style="font-family:\'IBM Plex Mono\'; font-size: 0.8rem; color: #8b949e;">DESEMPEÑO DEL CLASIFICADOR RANDOM FOREST (300 ESTIMADORES)</p>', unsafe_allow_html=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Overall Accuracy", "79.08%")
    c_m2.metric("Kappa Coefficient", "0.7384")
    c_m3.metric("Recall (Clase Manglar)", "98.83%")

    # Gráfico de Importancia de Variables
    df_imp = pd.DataFrame(list(BAND_IMPORTANCE.items()), columns=["Feature", "Importance"])
    df_imp = df_imp.sort_values("Importance", ascending=True).tail(12)  # Mostrar top 12

    fig_imp = go.Figure(go.Bar(
        x=df_imp["Importance"], y=df_imp["Feature"],
        orientation="h",
        marker=dict(color=df_imp["Importance"], colorscale="greens"),
        text=df_imp["Importance"].apply(lambda x: f"{x:.0f}"),
        textposition="outside",
        textfont=dict(color="white")
    ))
    fig_imp.update_layout(
        title="Top 12 Variables de Mayor Importancia (Gini Impurity)",
        title_font=dict(family="IBM Plex Mono", size=13, color="#8b949e"),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(family="IBM Plex Sans", color="#8b949e"),
        margin=dict(t=40, b=20, l=100, r=40), height=380,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 4: CHATBOT
# ────────────────────────────────────────────────────────────
with tab_chat:
    col_q, col_chat = st.columns([1, 2.5])

    with col_q:
        st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#8b949e;letter-spacing:0.1em;margin-bottom:12px">PREGUNTAS SUGERIDAS</p>', unsafe_allow_html=True)
        preguntas = [
            "Resumen del estado del manglar en 2025",
            "¿Cuánto manglar se perdió entre 2022 y 2025?",
            "¿Qué bandas del satélite fueron más útiles?",
            "¿Cuáles son las principales amenazas en Guayaquil?",
            "Explica la precisión del modelo Random Forest",
        ]
        for p in preguntas:
            if st.button(p, use_container_width=True):
                procesar_mensaje(p)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Limpiar conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_chat:
        chat_container = st.container(height=450)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:350px;color:#484f58;">
                  <div style="font-size:2.8rem;margin-bottom:16px">🌿</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;text-align:center;line-height:2">
                    ASISTENTE DE ECOLOGÍA COSTERA<br>
                    <span style="color:#3fb950">Haz una pregunta para explorar los datos satelitales</span>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                for m in st.session_state.messages:
                    content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", m["content"]).replace("\n", "<br>")
                    if m["role"] == "user":
                        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="msg-bot">{content}</div>', unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                user_in = st.text_input("", placeholder="Ej: ¿Cuáles son los motivos de pérdida de manglar?", label_visibility="collapsed")
            with c2:
                sent = st.form_submit_button("Enviar", use_container_width=True)
            if sent and user_in.strip():
                procesar_mensaje(user_in.strip())
