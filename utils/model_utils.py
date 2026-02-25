import pickle
import os
import numpy as np

MODELS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def cargar_modelo_svc(empresa):
    """Carga el modelo SVC y su scaler para una empresa."""
    modelo_path = os.path.join(MODELS_BASE, 'svc', f'svc_{empresa}.pkl')
    scaler_path = os.path.join(MODELS_BASE, 'svc', f'scaler_svc_{empresa}.pkl')

    if not os.path.exists(modelo_path):
        return None, None

    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return modelo, scaler


def cargar_modelo_dl(nombre_modelo, empresa):
    """Carga un modelo de Deep Learning (.h5) y su scaler."""
    modelo_path = os.path.join(MODELS_BASE, 'dl_clasificacion', f'{nombre_modelo.lower()}_{empresa}.h5')
    scaler_path = os.path.join(MODELS_BASE, 'dl_clasificacion', f'scaler_{empresa}.pkl')

    if not os.path.exists(modelo_path):
        return None, None

    try:
        from tensorflow.keras.models import load_model
        modelo = load_model(modelo_path)
    except ImportError:
        return None, None

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return modelo, scaler


def cargar_modelo_regresion(nombre_modelo, empresa):
    """Carga modelos de regresión (ARIMA, LSTM Regressor, ARIMA-LSTM)."""
    base = os.path.join(MODELS_BASE, 'regresion')

    if nombre_modelo == 'arima':
        path = os.path.join(base, f'arima_{empresa}.pkl')
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    elif nombre_modelo == 'lstm_regressor':
        modelo_path = os.path.join(base, f'lstm_regressor_{empresa}.h5')
        scaler_path = os.path.join(base, f'scaler_lstm_reg_{empresa}.pkl')
        if not os.path.exists(modelo_path):
            return None
        try:
            from tensorflow.keras.models import load_model
            modelo = load_model(modelo_path)
        except ImportError:
            return None
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return {'modelo': modelo, 'scaler': scaler}

    elif nombre_modelo == 'arima_lstm':
        config_path = os.path.join(base, f'arima_lstm_{empresa}.pkl')
        lstm_path = os.path.join(base, f'lstm_ensemble_{empresa}.h5')
        scaler_path = os.path.join(base, f'scaler_ensemble_{empresa}.pkl')
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        try:
            from tensorflow.keras.models import load_model
            lstm = load_model(lstm_path)
        except ImportError:
            return None
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return {'config': config, 'lstm': lstm, 'scaler': scaler}

    return None


def listar_modelos_disponibles():
    """Verifica qué modelos existen en disco."""
    disponibles = {'svc': [], 'dl': {}, 'regresion': {}}

    svc_dir = os.path.join(MODELS_BASE, 'svc')
    if os.path.exists(svc_dir):
        for f in os.listdir(svc_dir):
            if f.startswith('svc_') and f.endswith('.pkl'):
                empresa = f.replace('svc_', '').replace('.pkl', '')
                disponibles['svc'].append(empresa)

    dl_dir = os.path.join(MODELS_BASE, 'dl_clasificacion')
    modelos_dl = ['simplernn', 'lstm_classifier', 'bilstm', 'gru']
    if os.path.exists(dl_dir):
        for modelo in modelos_dl:
            disponibles['dl'][modelo] = []
            for f in os.listdir(dl_dir):
                if f.startswith(f'{modelo}_') and f.endswith('.h5'):
                    empresa = f.replace(f'{modelo}_', '').replace('.h5', '')
                    disponibles['dl'][modelo].append(empresa)

    reg_dir = os.path.join(MODELS_BASE, 'regresion')
    modelos_reg = {'arima': '.pkl', 'lstm_regressor': '.h5', 'arima_lstm': '.pkl'}
    if os.path.exists(reg_dir):
        for modelo, ext in modelos_reg.items():
            disponibles['regresion'][modelo] = []
            prefix = f'{modelo}_' if modelo != 'arima_lstm' else 'arima_lstm_'
            for f in os.listdir(reg_dir):
                if f.startswith(prefix) and f.endswith(ext):
                    empresa = f.replace(prefix, '').replace(ext, '')
                    disponibles['regresion'][modelo].append(empresa)

    return disponibles
