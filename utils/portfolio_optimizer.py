"""
Optimización de portafolio usando la Frontera Eficiente de Markowitz.
Calcula pesos óptimos para maximizar Sharpe Ratio o minimizar riesgo.
"""
import numpy as np
import pandas as pd
from utils.data_loader import obtener_datos_historicos


def obtener_retornos(empresas_tickers, start='2022-01-01', end=None):
    """Descarga datos y calcula retornos diarios para múltiples empresas."""
    if end is None:
        end = pd.Timestamp.now().strftime('%Y-%m-%d')

    precios = pd.DataFrame()
    for nombre, ticker in empresas_tickers.items():
        df = obtener_datos_historicos(ticker, start, end)
        if not df.empty:
            df = df.set_index('Date')
            precios[nombre] = df['Close']

    retornos = precios.pct_change().dropna()
    return precios, retornos


def calcular_metricas_portafolio(pesos, retornos):
    """Calcula retorno anual, volatilidad y Sharpe Ratio de un portafolio."""
    pesos = np.array(pesos)
    ret_anual = float(np.sum(retornos.mean() * pesos) * 252)
    vol_anual = float(np.sqrt(np.dot(pesos.T, np.dot(retornos.cov() * 252, pesos))))
    sharpe = ret_anual / vol_anual if vol_anual > 0 else 0.0
    return ret_anual, vol_anual, sharpe


def simulacion_montecarlo(retornos, n_portafolios=5000):
    """Genera portafolios aleatorios para trazar la frontera eficiente."""
    n_activos = len(retornos.columns)
    resultados = np.zeros((n_portafolios, 3 + n_activos))

    np.random.seed(42)
    for i in range(n_portafolios):
        pesos = np.random.random(n_activos)
        pesos /= pesos.sum()

        ret, vol, sharpe = calcular_metricas_portafolio(pesos, retornos)
        resultados[i, 0] = ret
        resultados[i, 1] = vol
        resultados[i, 2] = sharpe
        resultados[i, 3:] = pesos

    columnas = ['Retorno', 'Volatilidad', 'Sharpe'] + list(retornos.columns)
    df_resultados = pd.DataFrame(resultados, columns=columnas)
    return df_resultados


def portafolio_max_sharpe(df_simulacion, activos):
    """Encuentra el portafolio con mayor Sharpe Ratio."""
    idx = df_simulacion['Sharpe'].idxmax()
    fila = df_simulacion.loc[idx]
    pesos = {a: float(fila[a]) for a in activos}
    return {
        'retorno': float(fila['Retorno']),
        'volatilidad': float(fila['Volatilidad']),
        'sharpe': float(fila['Sharpe']),
        'pesos': pesos,
    }


def portafolio_min_riesgo(df_simulacion, activos):
    """Encuentra el portafolio con menor volatilidad."""
    idx = df_simulacion['Volatilidad'].idxmin()
    fila = df_simulacion.loc[idx]
    pesos = {a: float(fila[a]) for a in activos}
    return {
        'retorno': float(fila['Retorno']),
        'volatilidad': float(fila['Volatilidad']),
        'sharpe': float(fila['Sharpe']),
        'pesos': pesos,
    }


def recomendar_por_perfil(max_sharpe, min_riesgo, perfil='moderado'):
    """Sugiere un portafolio según el perfil de riesgo."""
    if perfil == 'conservador':
        return min_riesgo, 'Portafolio de mínimo riesgo'
    elif perfil == 'agresivo':
        return max_sharpe, 'Portafolio de máximo Sharpe'
    else:
        pesos_blend = {}
        for key in max_sharpe['pesos']:
            pesos_blend[key] = 0.5 * max_sharpe['pesos'][key] + 0.5 * min_riesgo['pesos'][key]
        total = sum(pesos_blend.values())
        pesos_blend = {k: v / total for k, v in pesos_blend.items()}
        ret = 0.5 * max_sharpe['retorno'] + 0.5 * min_riesgo['retorno']
        vol = 0.5 * max_sharpe['volatilidad'] + 0.5 * min_riesgo['volatilidad']
        sharpe = ret / vol if vol > 0 else 0
        return {
            'retorno': ret, 'volatilidad': vol,
            'sharpe': sharpe, 'pesos': pesos_blend,
        }, 'Portafolio balanceado (50% Sharpe + 50% Min. Riesgo)'
