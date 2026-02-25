EMPRESAS = {
    'FSM': {'nombre': 'Fortuna Silver Mines', 'ticker': 'FSM', 'bolsa': 'NYSE', 'pais': 'Canadá'},
    'VOLCABC1': {'nombre': 'Volcan Compañía Minera', 'ticker': 'VOLCABC1.LM', 'bolsa': 'BVL', 'pais': 'Perú'},
    'BVN': {'nombre': 'Buenaventura', 'ticker': 'BVN', 'bolsa': 'NYSE', 'pais': 'Perú'},
    'ABX': {'nombre': 'Barrick Gold', 'ticker': 'ABX.TO', 'bolsa': 'TSX', 'pais': 'Canadá'},
    'BHP': {'nombre': 'BHP Billiton', 'ticker': 'BHP', 'bolsa': 'NYSE', 'pais': 'Australia'},
    'SCCO': {'nombre': 'Southern Copper', 'ticker': 'SCCO', 'bolsa': 'NYSE', 'pais': 'USA'},
}

FEATURE_COLS = [
    'Return', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
    'RSI', 'BB_Width', 'Volatility_10', 'Volatility_20',
    'Volume_Ratio'
]

VENTANA_DL = 60
