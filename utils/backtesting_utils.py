"""
Motor de backtesting para simular estrategias de trading.
Calcula métricas: retorno total, Sharpe Ratio, Max Drawdown, Win Rate.
"""
import numpy as np
import pandas as pd


def _calcular_metricas(equity, trades, capital_inicial):
    """Calcula métricas de rendimiento a partir de la curva de equity."""
    retorno_total = ((equity.iloc[-1] / capital_inicial) - 1) * 100

    returns = equity.pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() != 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min() * 100)

    trades_cerrados = [t for t in trades if t.get('pnl') is not None]
    wins = sum(1 for t in trades_cerrados if t['pnl'] > 0)
    win_rate = (wins / len(trades_cerrados) * 100) if trades_cerrados else 0.0

    pnl_total = sum(t['pnl'] for t in trades_cerrados)

    return {
        'retorno_total': float(retorno_total),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_trades': len(trades_cerrados),
        'pnl_total': float(pnl_total),
        'capital_final': float(equity.iloc[-1]),
    }


def estrategia_buy_hold(df, capital=10000.0):
    """Estrategia benchmark: comprar al inicio y mantener."""
    precios = df['Close'].values
    acciones = capital / precios[0]
    equity = pd.Series(acciones * precios, index=df.index)

    trades = [{
        'tipo': 'compra', 'fecha': df['Date'].iloc[0],
        'precio': float(precios[0]), 'cantidad': float(acciones),
        'pnl': float((precios[-1] - precios[0]) * acciones),
    }]

    return equity, trades, _calcular_metricas(equity, trades, capital)


def estrategia_sma_crossover(df, sma_corta=20, sma_larga=50, capital=10000.0):
    """Comprar cuando SMA corta cruza arriba de SMA larga, vender al cruce inverso."""
    precios = df['Close'].values
    fechas = df['Date'].values
    sma_c = pd.Series(precios).rolling(sma_corta).mean().values
    sma_l = pd.Series(precios).rolling(sma_larga).mean().values

    cash = capital
    acciones = 0.0
    equity_list = []
    trades = []
    posicion_abierta = None

    for i in range(len(precios)):
        if np.isnan(sma_c[i]) or np.isnan(sma_l[i]):
            equity_list.append(cash + acciones * precios[i])
            continue

        if sma_c[i] > sma_l[i] and (i == 0 or sma_c[i - 1] <= sma_l[i - 1]) and acciones == 0:
            acciones = cash / precios[i]
            posicion_abierta = {'precio_compra': float(precios[i]), 'fecha_compra': fechas[i], 'cantidad': float(acciones)}
            trades.append({'tipo': 'compra', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones)})
            cash = 0.0

        elif sma_c[i] < sma_l[i] and (i == 0 or sma_c[i - 1] >= sma_l[i - 1]) and acciones > 0:
            cash = acciones * precios[i]
            pnl = (precios[i] - posicion_abierta['precio_compra']) * acciones
            trades.append({'tipo': 'venta', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones), 'pnl': float(pnl)})
            acciones = 0.0
            posicion_abierta = None

        equity_list.append(cash + acciones * precios[i])

    if acciones > 0 and posicion_abierta:
        pnl = (precios[-1] - posicion_abierta['precio_compra']) * acciones
        trades.append({'tipo': 'venta (cierre)', 'fecha': fechas[-1], 'precio': float(precios[-1]), 'cantidad': float(acciones), 'pnl': float(pnl)})

    equity = pd.Series(equity_list, index=df.index)
    return equity, trades, _calcular_metricas(equity, trades, capital)


def estrategia_rsi(df, periodo=14, sobreventa=30, sobrecompra=70, capital=10000.0):
    """Comprar cuando RSI < sobreventa, vender cuando RSI > sobrecompra."""
    precios = df['Close'].values
    fechas = df['Date'].values

    delta = pd.Series(precios).diff()
    gain = delta.where(delta > 0, 0).rolling(periodo).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).values

    cash = capital
    acciones = 0.0
    equity_list = []
    trades = []
    posicion_abierta = None

    for i in range(len(precios)):
        if np.isnan(rsi[i]):
            equity_list.append(cash + acciones * precios[i])
            continue

        if rsi[i] < sobreventa and acciones == 0:
            acciones = cash / precios[i]
            posicion_abierta = {'precio_compra': float(precios[i]), 'fecha_compra': fechas[i], 'cantidad': float(acciones)}
            trades.append({'tipo': 'compra', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones)})
            cash = 0.0

        elif rsi[i] > sobrecompra and acciones > 0:
            cash = acciones * precios[i]
            pnl = (precios[i] - posicion_abierta['precio_compra']) * acciones
            trades.append({'tipo': 'venta', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones), 'pnl': float(pnl)})
            acciones = 0.0
            posicion_abierta = None

        equity_list.append(cash + acciones * precios[i])

    if acciones > 0 and posicion_abierta:
        pnl = (precios[-1] - posicion_abierta['precio_compra']) * acciones
        trades.append({'tipo': 'venta (cierre)', 'fecha': fechas[-1], 'precio': float(precios[-1]), 'cantidad': float(acciones), 'pnl': float(pnl)})

    equity = pd.Series(equity_list, index=df.index)
    return equity, trades, _calcular_metricas(equity, trades, capital)


def estrategia_macd(df, rapida=12, lenta=26, signal=9, capital=10000.0):
    """Comprar cuando MACD cruza arriba de la señal, vender al cruce inverso."""
    precios = df['Close'].values
    fechas = df['Date'].values

    ema_r = pd.Series(precios).ewm(span=rapida).mean().values
    ema_l = pd.Series(precios).ewm(span=lenta).mean().values
    macd = ema_r - ema_l
    macd_signal = pd.Series(macd).ewm(span=signal).mean().values

    cash = capital
    acciones = 0.0
    equity_list = []
    trades = []
    posicion_abierta = None

    for i in range(len(precios)):
        if np.isnan(macd_signal[i]) or i == 0:
            equity_list.append(cash + acciones * precios[i])
            continue

        if macd[i] > macd_signal[i] and macd[i - 1] <= macd_signal[i - 1] and acciones == 0:
            acciones = cash / precios[i]
            posicion_abierta = {'precio_compra': float(precios[i]), 'fecha_compra': fechas[i], 'cantidad': float(acciones)}
            trades.append({'tipo': 'compra', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones)})
            cash = 0.0

        elif macd[i] < macd_signal[i] and macd[i - 1] >= macd_signal[i - 1] and acciones > 0:
            cash = acciones * precios[i]
            pnl = (precios[i] - posicion_abierta['precio_compra']) * acciones
            trades.append({'tipo': 'venta', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones), 'pnl': float(pnl)})
            acciones = 0.0
            posicion_abierta = None

        equity_list.append(cash + acciones * precios[i])

    if acciones > 0 and posicion_abierta:
        pnl = (precios[-1] - posicion_abierta['precio_compra']) * acciones
        trades.append({'tipo': 'venta (cierre)', 'fecha': fechas[-1], 'precio': float(precios[-1]), 'cantidad': float(acciones), 'pnl': float(pnl)})

    equity = pd.Series(equity_list, index=df.index)
    return equity, trades, _calcular_metricas(equity, trades, capital)


def estrategia_modelo_svc(df, empresa, capital=10000.0):
    """Backtesting usando señales del modelo SVC entrenado."""
    from utils.model_utils import cargar_modelo_svc
    from utils.data_loader import crear_features
    from config import FEATURE_COLS

    modelo, scaler = cargar_modelo_svc(empresa)
    if modelo is None:
        return None, None, None

    df_feat = crear_features(df.copy())
    if df_feat.empty:
        return None, None, None

    X = df_feat[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    predicciones = modelo.predict(X_scaled)

    precios = df_feat['Close'].values
    fechas = df_feat['Date'].values if 'Date' in df_feat.columns else df_feat.index.values

    cash = capital
    acciones = 0.0
    equity_list = []
    trades = []
    posicion_abierta = None

    for i in range(len(precios)):
        if predicciones[i] == 1 and acciones == 0:
            acciones = cash / precios[i]
            posicion_abierta = {'precio_compra': float(precios[i]), 'fecha_compra': fechas[i], 'cantidad': float(acciones)}
            trades.append({'tipo': 'compra', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones)})
            cash = 0.0

        elif predicciones[i] == 0 and acciones > 0:
            cash = acciones * precios[i]
            pnl = (precios[i] - posicion_abierta['precio_compra']) * acciones
            trades.append({'tipo': 'venta', 'fecha': fechas[i], 'precio': float(precios[i]), 'cantidad': float(acciones), 'pnl': float(pnl)})
            acciones = 0.0
            posicion_abierta = None

        equity_list.append(cash + acciones * precios[i])

    if acciones > 0 and posicion_abierta:
        pnl = (precios[-1] - posicion_abierta['precio_compra']) * acciones
        trades.append({'tipo': 'venta (cierre)', 'fecha': fechas[-1], 'precio': float(precios[-1]), 'cantidad': float(acciones), 'pnl': float(pnl)})

    equity = pd.Series(equity_list, index=df_feat.index)
    return equity, trades, _calcular_metricas(equity, trades, capital)
