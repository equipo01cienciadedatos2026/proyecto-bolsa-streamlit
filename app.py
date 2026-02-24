import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import date

st.set_page_config(page_title="Predicción Bolsa - MVP", layout="wide")

# CSS mejorado para mejor visualización de títulos
st.markdown("""
<style>
.stApp { 
    background-color: #ffffff; 
}

/* Mejorar títulos principales */
h1 {
    color: #1e3a8a !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Mejorar subtítulos */
h2, h3 {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 1.75rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
}

/* Mejorar contraste de labels */
label {
    color: #334155 !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Prototipo BI + IA aplicada a Bolsa (MVP)")

TICKERS = {
    "Fermín (FSM)": "FSM",
    "Volcan (VOLCABC1.LM)": "VOLCABC1.LM",
    "Buenaventura (BVN)": "BVN",
    "Barrick Gold (ABX)": "ABX",
    "BHP (BHP)": "BHP",
    "Southern Copper (SCCO)": "SCCO",
}

col1, col2, col3 = st.columns(3)

with col1:
    ticker_label = st.selectbox("Selecciona un activo", list(TICKERS.keys()))
    ticker = TICKERS[ticker_label]

with col2:
    start = st.date_input("Fecha inicio", value=date(2020, 1, 1))

with col3:
    end = st.date_input("Fecha fin", value=date.today())

@st.cache_data(ttl=3600)
def load_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

    # Por si yfinance devuelve columnas MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    return df

if st.button("Cargar datos"):
    if start >= end:
        st.error("La fecha inicio debe ser menor que la fecha fin.")
    else:
        df = load_data(ticker, str(start), str(end))
        if df.empty:
            st.warning("No se encontraron datos para ese rango.")
        else:
            st.success(f"Datos cargados: {len(df)} filas para {ticker}")
            st.subheader("Tabla de datos (OHLCV)")
            st.dataframe(df, use_container_width=True)

            st.subheader("Gráfico de cierre (Close)")
            fig = px.line(df, x="Date", y="Close", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)