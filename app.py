import streamlit as st

st.set_page_config(
    page_title='FinPredict - Predicción Bursátil con IA',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
/* ══════════════════════════════════════════
   PALETA: Azul marino + acentos esmeralda
   Primario:  #0f2b46  (azul marino profundo)
   Secundario:#1a7f64  (esmeralda)
   Acento:    #f0b429  (dorado)
   Superficie:#f7f9fc  (gris azulado muy claro)
   ══════════════════════════════════════════ */

/* === BASE === */
.stApp {
    background-color: #f7f9fc !important;
    color: #1e293b !important;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2b46 0%, #163a5c 100%) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
.sidebar-brand {
    text-align: center;
    padding: 0.8rem 0 0.3rem 0;
}
.sidebar-brand h2 {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.sidebar-brand p {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    margin: 4px 0 0 0 !important;
}
.sidebar-footer {
    text-align: center;
    padding: 0.6rem 0;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 1rem;
}
.sidebar-footer p {
    color: #64748b !important;
    font-size: 0.7rem !important;
    margin: 2px 0 !important;
}

/* === TÍTULOS === */
h1 {
    color: #0f2b46 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
h2 {
    color: #0f2b46 !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
h3 {
    color: #334155 !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}
.page-header {
    margin-bottom: 0.3rem;
}
.page-header h1 {
    margin-bottom: 0 !important;
}
.page-subtitle {
    color: #64748b !important;
    font-size: 0.92rem !important;
    margin-top: 0 !important;
}

/* === TEXTOS === */
p, span, div, li, td, th {
    color: #1e293b !important;
}

/* === MÉTRICAS === */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1a7f64;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(15,43,70,0.04);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 20px rgba(15,43,70,0.1);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] p {
    color: #0f2b46 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.35rem !important;
}
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* === BOTONES === */
.stButton > button {
    background-color: #ffffff !important;
    color: #0f2b46 !important;
    border: 1.5px solid #0f2b46 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    background-color: #0f2b46 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(15,43,70,0.3);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #1a7f64 !important;
    color: #ffffff !important;
    border: 1.5px solid #1a7f64 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #145f4b !important;
    color: #ffffff !important;
    border-color: #145f4b !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[data-testid="stBaseButton-primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] span {
    color: #ffffff !important;
}
.stButton > button:active,
.stButton > button:focus {
    color: #0f2b46 !important;
}
.stButton > button[kind="primary"]:active,
.stButton > button[kind="primary"]:focus,
.stButton > button[data-testid="stBaseButton-primary"]:active,
.stButton > button[data-testid="stBaseButton-primary"]:focus {
    color: #ffffff !important;
    background-color: #1a7f64 !important;
}

/* === INPUTS === */
[data-testid="stSelectbox"] label,
[data-testid="stDateInput"] label,
[data-testid="stNumberInput"] label {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #ffffff;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] * {
    color: #64748b !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #0f2b46 !important;
    border-bottom: none !important;
}
.stTabs [aria-selected="true"] * {
    color: #ffffff !important;
}

/* === EXPANDER === */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* === INFO/WARNING BOXES === */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* === LABELS === */
label {
    color: #334155 !important;
    font-weight: 600 !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* === DATAFRAME === */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}

/* === DIVIDER === */
hr {
    border-color: #e2e8f0 !important;
    margin: 1rem 0 !important;
}

/* === CARD CUSTOM === */
.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #1a7f64;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(15,43,70,0.04);
    transition: all 0.25s ease;
    height: 100%;
}
.info-card:hover {
    box-shadow: 0 8px 30px rgba(15,43,70,0.1);
    transform: translateY(-3px);
}
.info-card h3 {
    color: #0f2b46 !important;
    font-size: 1.1rem !important;
    margin-bottom: 8px !important;
}
.info-card p {
    color: #64748b !important;
    font-size: 0.88rem !important;
    line-height: 1.5;
}
.card-icon {
    font-size: 2rem;
    margin-bottom: 10px;
}

/* === BADGE === */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-green {
    background-color: #dcfce7;
    color: #166534 !important;
}
.badge-red {
    background-color: #fee2e2;
    color: #991b1b !important;
}
.badge-blue {
    background-color: #dbeafe;
    color: #0f2b46 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>📈 FinPredict</h2>
        <p>Sistema de Predicción Bursátil con IA</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    pagina = st.radio(
        'Navegación',
        [
            '🏠 Dashboard',
            '🔮 Clasificación',
            '📊 Regresión',
            '📈 Backtesting',
            '💼 Portafolio',
            '🗄️ Base de Datos',
        ],
        label_visibility='collapsed'
    )

    st.markdown("""
    <div class="sidebar-footer">
        <p>UNMSM — Inteligencia de Negocios</p>
        <p>Semestre 2026-0</p>
        <p style="margin-top:6px;">v1.0 MVP</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Enrutamiento ───
if pagina == '🏠 Dashboard':
    from views.dashboard import render
    render()

elif pagina == '🔮 Clasificación':
    from views.clasificacion import render as render_cls
    render_cls()

elif pagina == '📊 Regresión':
    from views.regresion import render as render_reg
    render_reg()

elif pagina == '📈 Backtesting':
    from views.backtesting import render as render_bt
    render_bt()

elif pagina == '💼 Portafolio':
    from views.portafolio import render as render_port
    render_port()

elif pagina == '🗄️ Base de Datos':
    from views.base_datos import render as render_bd
    render_bd()
