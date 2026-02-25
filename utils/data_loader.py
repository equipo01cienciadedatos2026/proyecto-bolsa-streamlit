import streamlit as st
import yfinance as yf
import pandas as pd
from config import EMPRESAS


@st.cache_data(ttl=900)
def obtener_precios_actuales():
    resultados = {}
    for key, info in EMPRESAS.items():
        try:
            data = yf.download(info['ticker'], period='5d', auto_adjust=False, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if len(data) >= 2:
                precio = float(data['Close'].iloc[-1])
                anterior = float(data['Close'].iloc[-2])
                cambio = ((precio / anterior) - 1) * 100
                volumen = float(data['Volume'].iloc[-1])
                resultados[key] = {
                    'precio': precio,
                    'cambio': cambio,
                    'volumen': volumen,
                    'nombre': info['nombre']
                }
            else:
                resultados[key] = None
        except Exception:
            resultados[key] = None
    return resultados


@st.cache_data(ttl=3600)
def obtener_datos_historicos(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df


def crear_features(df):
    """Crea indicadores técnicos para los modelos de predicción."""
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    bb_sma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Width'] = (4 * bb_std) / bb_sma
    df['Volatility_10'] = df['Return'].rolling(10).std()
    df['Volatility_20'] = df['Return'].rolling(20).std()
    df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
    df = df.dropna()
    return df
