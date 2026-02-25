import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from config import EMPRESAS, FEATURE_COLS, VENTANA_DL
from utils.data_loader import obtener_datos_historicos, crear_features
from utils.model_utils import MODELS_BASE, listar_modelos_disponibles
from database.db_utils import guardar_prediccion

TF_DISPONIBLE = False
try:
    import tensorflow
    TF_DISPONIBLE = True
except ImportError:
    pass

MODELOS_REG = {
    'ARIMA': {
        'key': 'arima',
        'desc': 'Autoregressive Integrated Moving Average — Modelo estadístico clásico para series temporales',
        'necesita_tf': False,
    },
    'LSTM Regressor': {
        'key': 'lstm_regressor',
        'desc': 'Long Short-Term Memory para regresión — Red neuronal que predice precios futuros',
        'necesita_tf': True,
    },
    'ARIMA-LSTM': {
        'key': 'arima_lstm',
        'desc': 'Ensamblaje ARIMA + LSTM — Combina la predicción estadística con Deep Learning',
        'necesita_tf': True,
    },
}

CHART_LAYOUT = dict(
    template='plotly_white',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#334155', family='sans-serif', size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    hoverlabel=dict(bgcolor='#0f2b46', font_color='white', font_size=13, bordercolor='#0f2b46'),
)


def _predecir_arima(empresa, dias=5):
    path = os.path.join(MODELS_BASE, 'regresion', f'arima_{empresa}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    modelo = data['model']
    forecast = modelo.forecast(steps=dias)
    return {'precios': list(forecast), 'modo': 'real', 'order': data.get('order')}


def _predecir_lstm_reg(empresa, df_features, dias=5):
    modelo_path = os.path.join(MODELS_BASE, 'regresion', f'lstm_regressor_{empresa}.h5')
    scaler_path = os.path.join(MODELS_BASE, 'regresion', f'scaler_lstm_reg_{empresa}.pkl')
    if not os.path.exists(modelo_path):
        return None
    from tensorflow.keras.models import load_model
    modelo = load_model(modelo_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    close_vals = df_features['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_vals)
    window = scaled[-VENTANA_DL:]

    precios = []
    current = window.copy()
    for _ in range(dias):
        X = current.reshape(1, VENTANA_DL, 1)
        pred_scaled = modelo.predict(X, verbose=0)[0][0]
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        precios.append(float(pred_real))
        current = np.append(current[1:], [[pred_scaled]], axis=0)

    return {'precios': precios, 'modo': 'real'}


def _predecir_ensemble(empresa, df_features, dias=5):
    config_path = os.path.join(MODELS_BASE, 'regresion', f'arima_lstm_{empresa}.pkl')
    lstm_path = os.path.join(MODELS_BASE, 'regresion', f'lstm_ensemble_{empresa}.h5')
    scaler_path = os.path.join(MODELS_BASE, 'regresion', f'scaler_ensemble_{empresa}.pkl')
    if not os.path.exists(config_path):
        return None

    arima_result = _predecir_arima(empresa, dias)
    if arima_result is None:
        return None

    from tensorflow.keras.models import load_model
    lstm = load_model(lstm_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    close_vals = df_features['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_vals)
    window = scaled[-VENTANA_DL:]

    lstm_precios = []
    current = window.copy()
    for _ in range(dias):
        X = current.reshape(1, VENTANA_DL, 1)
        pred_scaled = lstm.predict(X, verbose=0)[0][0]
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        lstm_precios.append(float(pred_real))
        current = np.append(current[1:], [[pred_scaled]], axis=0)

    ensemble = [(a + l) / 2 for a, l in zip(arima_result['precios'], lstm_precios)]
    return {'precios': ensemble, 'modo': 'real', 'arima': arima_result['precios'], 'lstm': lstm_precios}


def _predecir_lstm_demo(empresa, df_features, dias=5):
    """Demo para LSTM Regressor: extrapolación lineal con tendencia reciente."""
    precios = df_features['Close'].values
    if len(precios) < 20:
        return None

    recientes = precios[-20:]
    tendencia = (recientes[-1] - recientes[-5]) / 5
    volatilidad = np.std(np.diff(recientes[-10:])) * 0.5

    rng = np.random.RandomState(hash(empresa) % 10000)
    resultado = []
    ultimo = float(precios[-1])
    for i in range(dias):
        noise = rng.normal(0, volatilidad)
        ultimo = ultimo + tendencia * (0.8 ** i) + noise
        resultado.append(float(ultimo))

    return {'precios': resultado, 'modo': 'demo'}


def _predecir_ensemble_demo(empresa, df_features, dias=5):
    """Demo para ARIMA-LSTM: promedio de ARIMA real + extrapolación."""
    arima_res = _predecir_arima(empresa, dias)
    lstm_demo = _predecir_lstm_demo(empresa, df_features, dias)

    if arima_res is None or lstm_demo is None:
        return None

    ensemble = [(a + l) / 2 for a, l in zip(arima_res['precios'], lstm_demo['precios'])]
    return {
        'precios': ensemble,
        'modo': 'demo',
        'arima': arima_res['precios'],
        'lstm_demo': lstm_demo['precios'],
    }


def render():
    st.markdown('<div class="page-header"><h1>Modelos de Regresión</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Predicción de precio para los próximos días — 3 modelos: ARIMA, LSTM Regressor y Ensamblaje</p>',
        unsafe_allow_html=True
    )

    disponibles = listar_modelos_disponibles()

    col_sel, col_dias, col_info = st.columns([2.5, 1, 2])
    with col_sel:
        opciones = [f'{k} — {v["nombre"]}' for k, v in EMPRESAS.items()]
        seleccion = st.selectbox('EMPRESA', opciones, key='reg_empresa')
        empresa_key = seleccion.split(' — ')[0]
    with col_dias:
        dias_pred = st.number_input('Días a predecir', min_value=1, max_value=30, value=5, key='reg_dias')
    with col_info:
        info = EMPRESAS[empresa_key]
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin-top:1.5rem;">
            <span class="badge badge-blue">{info['bolsa']}</span>
            <span class="badge badge-blue">{info['ticker']}</span>
            <span style="color:#64748b; font-size:0.85rem;">{info['pais']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Estado de modelos ──
    modelo_disponible = {}
    for nombre, cfg in MODELOS_REG.items():
        key = cfg['key']
        empresa_list = disponibles.get('regresion', {}).get(key, [])
        modelo_disponible[nombre] = empresa_key in empresa_list

    total_ok = sum(modelo_disponible.values())
    if total_ok == 0:
        st.warning(f'No se encontraron modelos de regresión entrenados para **{empresa_key}**.')
        return

    if not TF_DISPONIBLE:
        modelos_tf = [n for n, c in MODELOS_REG.items() if c['necesita_tf'] and modelo_disponible[n]]
        if modelos_tf:
            st.info(
                f'**Modo Demo para {", ".join(modelos_tf)}:** TensorFlow no disponible en Python 3.14. '
                'ARIMA funciona con predicciones reales. Los modelos DL usarán extrapolación estadística.',
                icon='ℹ️'
            )

    cols_estado = st.columns(3)
    for i, (nombre, cfg) in enumerate(MODELOS_REG.items()):
        with cols_estado[i]:
            ok = modelo_disponible[nombre]
            necesita_tf = cfg['necesita_tf']
            if ok and necesita_tf and not TF_DISPONIBLE:
                color, icon, modo_txt = '#f59e0b', '⚡', 'Demo'
            elif ok:
                color, icon, modo_txt = '#16a34a', '✅', 'Listo'
            else:
                color, icon, modo_txt = '#dc2626', '❌', 'No disponible'
            st.markdown(
                f'<div style="text-align:center; padding:10px; background:#fff; border:1px solid #e2e8f0; '
                f'border-radius:8px; border-left:3px solid {color};">'
                f'<p style="margin:0; font-weight:700; font-size:0.95rem;">{icon} {nombre}</p>'
                f'<p style="margin:2px 0 0 0; font-size:0.72rem; color:#94a3b8 !important;">{modo_txt}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── Ejecutar predicción ──
    if st.button('Ejecutar predicción de precios', use_container_width=True, type='primary'):
        with st.spinner('Descargando datos históricos...'):
            df_raw = obtener_datos_historicos(info['ticker'], '2018-01-01', datetime.now().strftime('%Y-%m-%d'))
            if df_raw.empty or len(df_raw) < VENTANA_DL + 10:
                st.error('No hay datos suficientes.')
                return
            df_feat = crear_features(df_raw)

        precio_actual = float(df_raw['Close'].iloc[-1])
        fecha_ultimo = pd.to_datetime(df_raw['Date'].iloc[-1])
        st.markdown(
            f'<p style="color:#64748b; font-size:0.8rem; text-align:right;">'
            f'Último cierre: <b>${precio_actual:.2f}</b> ({fecha_ultimo.strftime("%d/%m/%Y")})</p>',
            unsafe_allow_html=True
        )

        resultados = {}

        with st.spinner('Ejecutando modelos de regresión...'):
            progress = st.progress(0)
            modelos_list = [(n, c) for n, c in MODELOS_REG.items() if modelo_disponible[n]]

            for idx, (nombre, cfg) in enumerate(modelos_list):
                try:
                    if cfg['key'] == 'arima':
                        res = _predecir_arima(empresa_key, dias_pred)
                    elif cfg['key'] == 'lstm_regressor':
                        if TF_DISPONIBLE:
                            res = _predecir_lstm_reg(empresa_key, df_feat, dias_pred)
                        else:
                            res = _predecir_lstm_demo(empresa_key, df_feat, dias_pred)
                    elif cfg['key'] == 'arima_lstm':
                        if TF_DISPONIBLE:
                            res = _predecir_ensemble(empresa_key, df_feat, dias_pred)
                        else:
                            res = _predecir_ensemble_demo(empresa_key, df_feat, dias_pred)
                    else:
                        res = None

                    if res:
                        resultados[nombre] = res
                    else:
                        resultados[nombre] = {'error': 'No se pudo ejecutar'}
                except Exception as e:
                    resultados[nombre] = {'error': str(e)}

                progress.progress((idx + 1) / len(modelos_list))
            progress.empty()

        if not resultados:
            st.error('No se pudieron ejecutar los modelos.')
            return

        # ── Fechas futuras ──
        fechas_futuras = []
        fecha_base = fecha_ultimo
        for _ in range(dias_pred):
            fecha_base += timedelta(days=1)
            while fecha_base.weekday() >= 5:
                fecha_base += timedelta(days=1)
            fechas_futuras.append(fecha_base)

        # ── Tarjetas de resultado ──
        st.markdown('### Predicciones de Precio')

        resultados_validos = {k: v for k, v in resultados.items() if 'precios' in v}

        cols_card = st.columns(len(resultados))
        for i, (nombre, res) in enumerate(resultados.items()):
            with cols_card[i]:
                if 'error' in res:
                    st.markdown(
                        f'<div style="text-align:center; padding:16px; background:#fff; border:1px solid #e2e8f0; '
                        f'border-radius:10px; border-top:3px solid #dc2626;">'
                        f'<p style="font-weight:700; margin:0;">{nombre}</p>'
                        f'<p style="color:#dc2626 !important; font-size:0.8rem;">{res["error"]}</p></div>',
                        unsafe_allow_html=True
                    )
                    continue

                precio_pred = res['precios'][-1]
                cambio = ((precio_pred / precio_actual) - 1) * 100
                color = '#16a34a' if cambio >= 0 else '#dc2626'
                arrow = '↑' if cambio >= 0 else '↓'
                bg = '#f0fdf4' if cambio >= 0 else '#fef2f2'
                modo = res.get('modo', 'real')
                demo_badge = (
                    '<span style="display:inline-block; background:#fef3c7; color:#92400e !important; '
                    'font-size:0.65rem; padding:2px 6px; border-radius:10px; font-weight:700; '
                    'margin-left:4px;">DEMO</span>'
                    if modo == 'demo' else ''
                )

                st.markdown(
                    f'<div style="text-align:center; padding:20px 14px; background:{bg}; '
                    f'border:1px solid #e2e8f0; border-radius:10px; border-top:3px solid {color};">'
                    f'<p style="font-weight:600; font-size:0.85rem; color:#64748b !important; margin:0 0 6px 0;">{nombre}{demo_badge}</p>'
                    f'<p style="font-size:1.5rem; font-weight:800; color:{color} !important; margin:0;">${precio_pred:.2f}</p>'
                    f'<p style="font-size:0.9rem; color:{color} !important; margin:4px 0 0 0;">{arrow} {cambio:+.2f}%</p>'
                    f'<p style="font-size:0.72rem; color:#94a3b8 !important; margin:4px 0 0 0;">'
                    f'a {dias_pred} días ({fechas_futuras[-1].strftime("%d/%m/%Y")})</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Gráfico de predicciones ──
        if resultados_validos:
            st.markdown('### Proyección de Precios')

            historico_dias = min(60, len(df_raw))
            df_hist = df_raw.tail(historico_dias)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df_hist['Date']),
                y=df_hist['Close'],
                mode='lines',
                name='Precio real',
                line=dict(color='#0f2b46', width=2.5),
            ))

            colores_modelo = {'ARIMA': '#1a7f64', 'LSTM Regressor': '#2563eb', 'ARIMA-LSTM': '#9333ea'}

            for nombre, res in resultados_validos.items():
                precios_full = [precio_actual] + res['precios']
                fechas_full = [fecha_ultimo] + fechas_futuras[:len(res['precios'])]
                modo = res.get('modo', 'real')
                dash = 'dash' if modo == 'demo' else 'solid'
                label = f'{nombre} (Demo)' if modo == 'demo' else nombre

                fig.add_trace(go.Scatter(
                    x=fechas_full,
                    y=precios_full,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=colores_modelo.get(nombre, '#64748b'), width=2, dash=dash),
                    marker=dict(size=5),
                ))

            fig.add_shape(
                type='line',
                x0=fecha_ultimo.strftime('%Y-%m-%d'), x1=fecha_ultimo.strftime('%Y-%m-%d'),
                y0=0, y1=1, yref='paper',
                line=dict(color='#94a3b8', dash='dot', width=1.5),
            )
            fig.add_annotation(
                x=fecha_ultimo.strftime('%Y-%m-%d'), y=1, yref='paper',
                text='Hoy', showarrow=False,
                font=dict(color='#64748b', size=11),
                yshift=10,
            )

            fig.update_layout(
                **CHART_LAYOUT,
                height=450,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                yaxis=dict(title='Precio (USD)', gridcolor='#eef2f6'),
                xaxis=dict(gridcolor='#eef2f6'),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Tabla día a día ──
        st.markdown('### Detalle Día a Día')
        tabla_data = {'Fecha': [f.strftime('%d/%m/%Y (%a)') for f in fechas_futuras]}
        for nombre, res in resultados_validos.items():
            modo_tag = ' (Demo)' if res.get('modo') == 'demo' else ''
            tabla_data[f'{nombre}{modo_tag}'] = [f'${p:.2f}' for p in res['precios']]

        df_tabla = pd.DataFrame(tabla_data)
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)

        # ── Métricas comparativas ──
        if len(resultados_validos) > 1:
            st.markdown('### Comparación de Modelos')
            cols_comp = st.columns(len(resultados_validos))
            for i, (nombre, res) in enumerate(resultados_validos.items()):
                with cols_comp[i]:
                    precios = res['precios']
                    precio_min = min(precios)
                    precio_max = max(precios)
                    precio_final = precios[-1]
                    cambio_total = ((precio_final / precio_actual) - 1) * 100
                    volatilidad = np.std(precios)

                    st.markdown(f'**{nombre}**')
                    st.metric('Precio final', f'${precio_final:.2f}', f'{cambio_total:+.2f}%')
                    st.metric('Rango', f'${precio_min:.2f} - ${precio_max:.2f}')
                    st.metric('Volatilidad', f'${volatilidad:.4f}')

        # ── Guardar en BD ──
        with st.expander('Guardar predicciones en base de datos'):
            if st.button('Guardar predicciones de regresión', key='guardar_reg'):
                guardadas = 0
                for nombre, res in resultados_validos.items():
                    precio_final = res['precios'][-1]
                    guardar_prediccion(
                        empresa=empresa_key,
                        modelo=nombre,
                        tipo='regresion',
                        resultado=f'${precio_final:.2f}',
                        confianza=None,
                        fecha_objetivo=fechas_futuras[-1].strftime('%Y-%m-%d')
                    )
                    guardadas += 1
                st.success(f'{guardadas} predicciones guardadas.')

    # ── Explicación ──
    st.divider()
    st.markdown('### Acerca de los Modelos')
    for nombre, cfg in MODELOS_REG.items():
        with st.expander(nombre):
            st.markdown(f'**{cfg["desc"]}**')
            if nombre == 'ARIMA':
                st.markdown("""
                - **Tipo:** Modelo estadístico (statsmodels/pmdarima)
                - **Qué hace:** Modela la serie temporal como combinación de sus valores pasados, diferencias y errores
                - **Ventaja:** No necesita GPU, rápido, bueno para tendencias a corto plazo
                - **Parámetros:** Optimizados automáticamente con `auto_arima` (p, d, q)
                """)
            elif nombre == 'LSTM Regressor':
                st.markdown(f"""
                - **Tipo:** Deep Learning (TensorFlow/Keras)
                - **Qué hace:** Usa una ventana de {VENTANA_DL} precios anteriores para predecir el siguiente
                - **Ventaja:** Captura patrones no lineales complejos
                - **Entrada:** Secuencia de precios de cierre normalizados
                """)
            else:
                st.markdown(f"""
                - **Tipo:** Ensamblaje (Estadístico + Deep Learning)
                - **Qué hace:** Promedia las predicciones de ARIMA y LSTM para mayor robustez
                - **Ventaja:** Reduce el error de cada modelo individual al combinarlos
                - **Fórmula:** Predicción = (ARIMA + LSTM) / 2
                """)
