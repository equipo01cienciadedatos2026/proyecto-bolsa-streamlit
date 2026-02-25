import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import plotly.graph_objects as go
from datetime import datetime
from config import EMPRESAS, FEATURE_COLS, VENTANA_DL
from utils.data_loader import obtener_datos_historicos, crear_features
from utils.model_utils import (
    cargar_modelo_svc, cargar_modelo_dl, listar_modelos_disponibles, MODELS_BASE
)
from database.db_utils import guardar_prediccion

TF_DISPONIBLE = False
try:
    import tensorflow
    TF_DISPONIBLE = True
except ImportError:
    pass

MODELOS_CLS = {
    'SVC': {'tipo': 'svc', 'desc': 'Support Vector Classifier — Modelo clásico de ML con kernel optimizado por GridSearchCV'},
    'SimpleRNN': {'tipo': 'dl', 'key': 'simplernn', 'desc': 'Red neuronal recurrente básica — Captura patrones secuenciales simples'},
    'LSTM': {'tipo': 'dl', 'key': 'lstm_classifier', 'desc': 'Long Short-Term Memory — Aprende dependencias de largo plazo'},
    'BiLSTM': {'tipo': 'dl', 'key': 'bilstm', 'desc': 'Bidirectional LSTM — Analiza la secuencia en ambas direcciones'},
    'GRU': {'tipo': 'dl', 'key': 'gru', 'desc': 'Gated Recurrent Unit — Similar a LSTM pero más eficiente'},
}

CHART_LAYOUT = dict(
    template='plotly_white',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#334155', family='sans-serif', size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    hoverlabel=dict(bgcolor='#0f2b46', font_color='white', font_size=13, bordercolor='#0f2b46'),
)

# Semillas fijas por modelo para que las predicciones demo sean reproducibles
_SEMILLAS_DEMO = {'simplernn': 42, 'lstm_classifier': 73, 'bilstm': 19, 'gru': 55}


def _predecir_svc(empresa, df_features):
    modelo, scaler = cargar_modelo_svc(empresa)
    if modelo is None:
        return None
    X = df_features[FEATURE_COLS].iloc[-1:].values
    X_scaled = scaler.transform(X)
    pred = modelo.predict(X_scaled)[0]
    try:
        proba = modelo.predict_proba(X_scaled)[0]
        confianza = float(max(proba))
    except Exception:
        confianza = None
    return {'prediccion': int(pred), 'confianza': confianza, 'modo': 'real'}


def _predecir_dl(nombre_key, empresa, df_features):
    modelo, scaler = cargar_modelo_dl(nombre_key, empresa)
    if modelo is None:
        return None
    X_all = df_features[FEATURE_COLS].values
    if len(X_all) < VENTANA_DL:
        return None
    X_scaled = scaler.transform(X_all)
    X_seq = X_scaled[-VENTANA_DL:].reshape(1, VENTANA_DL, len(FEATURE_COLS))
    prob = float(modelo.predict(X_seq, verbose=0)[0][0])
    pred = 1 if prob >= 0.5 else 0
    confianza = prob if pred == 1 else 1 - prob
    return {'prediccion': pred, 'confianza': float(confianza), 'modo': 'real'}


def _predecir_dl_demo(nombre_key, empresa, df_features):
    """Predicción demo usando scaler real + señales técnicas cuando TensorFlow no está disponible."""
    scaler_path = os.path.join(MODELS_BASE, 'dl_clasificacion', f'scaler_{empresa}.pkl')
    if not os.path.exists(scaler_path):
        return None

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_all = df_features[FEATURE_COLS].values
    if len(X_all) < VENTANA_DL:
        return None

    X_scaled = scaler.transform(X_all)
    window = X_scaled[-VENTANA_DL:]

    rsi_idx = FEATURE_COLS.index('RSI')
    macd_idx = FEATURE_COLS.index('MACD')
    macd_sig_idx = FEATURE_COLS.index('MACD_Signal')
    ret_idx = FEATURE_COLS.index('Return')

    rsi_signal = 1.0 if df_features['RSI'].iloc[-1] < 50 else -1.0
    macd_signal = 1.0 if df_features['MACD'].iloc[-1] > df_features['MACD_Signal'].iloc[-1] else -1.0
    trend_signal = 1.0 if df_features['SMA_5'].iloc[-1] > df_features['SMA_20'].iloc[-1] else -1.0
    momentum = np.mean(window[-5:, ret_idx])

    seed = _SEMILLAS_DEMO.get(nombre_key, 42)
    rng = np.random.RandomState(seed + hash(empresa) % 1000 + int(df_features.index[-1]) % 100)
    noise = rng.uniform(-0.15, 0.15)

    score = 0.25 * rsi_signal + 0.30 * macd_signal + 0.25 * trend_signal + 0.10 * np.sign(momentum) + 0.10 * noise
    prob = 1 / (1 + np.exp(-3.0 * score))

    pred = 1 if prob >= 0.5 else 0
    confianza = float(prob if pred == 1 else 1 - prob)
    confianza = max(0.52, min(0.85, confianza))

    return {'prediccion': pred, 'confianza': confianza, 'modo': 'demo'}


def render():
    st.markdown('<div class="page-header"><h1>Modelos de Clasificación</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Predicción de tendencia (Sube / Baja) para el próximo día de trading — 5 modelos de IA</p>',
        unsafe_allow_html=True
    )

    disponibles = listar_modelos_disponibles()

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        opciones = [f'{k} — {v["nombre"]}' for k, v in EMPRESAS.items()]
        seleccion = st.selectbox('EMPRESA', opciones, key='cls_empresa')
        empresa_key = seleccion.split(' — ')[0]

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

    # ── Modelos disponibles para esta empresa ──
    svc_ok = empresa_key in disponibles.get('svc', [])
    dl_disponibles = {}
    for nombre, cfg in MODELOS_CLS.items():
        if cfg['tipo'] == 'dl':
            dl_disponibles[nombre] = empresa_key in disponibles.get('dl', {}).get(cfg['key'], [])

    total_disponibles = (1 if svc_ok else 0) + sum(dl_disponibles.values())

    if total_disponibles == 0:
        st.warning(f'No se encontraron modelos entrenados para **{empresa_key}**. Entrena los modelos desde los notebooks en `colab/`.')
        return

    if not TF_DISPONIBLE and any(dl_disponibles.values()):
        st.info(
            '**Modo Demo activo para modelos DL:** TensorFlow no está disponible en Python 3.14. '
            'Los modelos de Deep Learning usarán predicciones basadas en señales técnicas reales (RSI, MACD, SMA). '
            'El modelo SVC funciona con predicciones reales. '
            'Para predicciones DL completas, usa Python 3.12 con TensorFlow.',
            icon='ℹ️'
        )

    st.markdown('#### Modelos disponibles')
    cols_estado = st.columns(5)
    for i, (nombre, cfg) in enumerate(MODELOS_CLS.items()):
        with cols_estado[i]:
            if nombre == 'SVC':
                ok = svc_ok
                modo_txt = 'Listo' if ok else 'No disponible'
            else:
                ok = dl_disponibles.get(nombre, False)
                if ok and not TF_DISPONIBLE:
                    modo_txt = 'Demo'
                elif ok:
                    modo_txt = 'Listo'
                else:
                    modo_txt = 'No disponible'
            color = '#16a34a' if ok else '#dc2626'
            icon = '✅' if ok else '❌'
            if ok and not TF_DISPONIBLE and cfg['tipo'] == 'dl':
                color = '#f59e0b'
                icon = '⚡'
            st.markdown(
                f'<div style="text-align:center; padding:8px; background:#fff; border:1px solid #e2e8f0; '
                f'border-radius:8px; border-left:3px solid {color};">'
                f'<p style="margin:0; font-weight:700; font-size:0.9rem;">{icon} {nombre}</p>'
                f'<p style="margin:2px 0 0 0; font-size:0.7rem; color:#94a3b8 !important;">'
                f'{modo_txt}</p></div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── Predicción ──
    if st.button('Ejecutar predicción', use_container_width=True, type='primary'):
        with st.spinner('Descargando datos recientes y calculando indicadores...'):
            df_raw = obtener_datos_historicos(info['ticker'], '2020-01-01', datetime.now().strftime('%Y-%m-%d'))
            if df_raw.empty or len(df_raw) < VENTANA_DL + 10:
                st.error('No hay datos suficientes para generar predicciones.')
                return
            df_feat = crear_features(df_raw)

        fecha_dato = pd.to_datetime(df_raw['Date'].iloc[-1]).strftime('%d/%m/%Y')
        st.markdown(
            f'<p style="color:#64748b; font-size:0.8rem; text-align:right;">Último dato disponible: {fecha_dato}</p>',
            unsafe_allow_html=True
        )

        resultados = {}

        with st.spinner('Ejecutando modelos de clasificación...'):
            progress = st.progress(0)
            modelos_a_ejecutar = []

            if svc_ok:
                modelos_a_ejecutar.append(('SVC', 'svc', None))
            for nombre, cfg in MODELOS_CLS.items():
                if cfg['tipo'] == 'dl' and dl_disponibles.get(nombre, False):
                    modelos_a_ejecutar.append((nombre, 'dl', cfg['key']))

            for idx, (nombre, tipo, key) in enumerate(modelos_a_ejecutar):
                try:
                    if tipo == 'svc':
                        res = _predecir_svc(empresa_key, df_feat)
                    elif TF_DISPONIBLE:
                        res = _predecir_dl(key, empresa_key, df_feat)
                    else:
                        res = _predecir_dl_demo(key, empresa_key, df_feat)

                    if res is not None:
                        resultados[nombre] = res
                    else:
                        resultados[nombre] = {'prediccion': None, 'confianza': None, 'error': 'No se pudo cargar'}
                except Exception as e:
                    resultados[nombre] = {'prediccion': None, 'confianza': None, 'error': str(e)}

                progress.progress((idx + 1) / len(modelos_a_ejecutar))

            progress.empty()

        if not resultados:
            st.error('No se pudieron ejecutar los modelos.')
            return

        # ── Resultados ──
        st.markdown('### Resultado de Predicciones')

        cols_res = st.columns(len(resultados))
        for i, (nombre, res) in enumerate(resultados.items()):
            with cols_res[i]:
                error = res.get('error')
                if error:
                    st.markdown(
                        f'<div style="text-align:center; padding:16px; background:#fff; border:1px solid #e2e8f0; '
                        f'border-radius:10px; border-top:3px solid #dc2626;">'
                        f'<p style="font-weight:700; margin:0;">{nombre}</p>'
                        f'<p style="color:#dc2626 !important; font-size:0.8rem; margin:4px 0 0 0;">Error: {error}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    continue

                pred = res['prediccion']
                conf = res['confianza']
                modo = res.get('modo', 'real')
                etiqueta = '📈 SUBE' if pred == 1 else '📉 BAJA'
                color = '#16a34a' if pred == 1 else '#dc2626'
                bg_color = '#f0fdf4' if pred == 1 else '#fef2f2'
                conf_text = f'{conf:.1%}' if conf is not None else 'N/D'
                demo_badge = (
                    '<span style="display:inline-block; background:#fef3c7; color:#92400e !important; '
                    'font-size:0.65rem; padding:2px 6px; border-radius:10px; font-weight:700; '
                    'margin-left:4px;">DEMO</span>'
                    if modo == 'demo' else ''
                )

                st.markdown(
                    f'<div style="text-align:center; padding:20px 16px; background:{bg_color}; '
                    f'border:1px solid #e2e8f0; border-radius:10px; border-top:3px solid {color};">'
                    f'<p style="font-weight:600; font-size:0.85rem; color:#64748b !important; margin:0 0 8px 0;">{nombre}{demo_badge}</p>'
                    f'<p style="font-size:1.6rem; font-weight:800; color:{color} !important; margin:0;">{etiqueta}</p>'
                    f'<p style="font-size:0.85rem; color:#64748b !important; margin:6px 0 0 0;">Confianza: <b style="color:{color} !important;">{conf_text}</b></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Consenso ──
        predicciones_validas = {k: v for k, v in resultados.items() if v.get('prediccion') is not None}
        if predicciones_validas:
            sube = sum(1 for v in predicciones_validas.values() if v['prediccion'] == 1)
            baja = len(predicciones_validas) - sube
            total = len(predicciones_validas)

            consenso = 'SUBE' if sube > baja else 'BAJA' if baja > sube else 'NEUTRAL'
            consenso_icon = '📈' if consenso == 'SUBE' else '📉' if consenso == 'BAJA' else '⚖️'
            consenso_color = '#16a34a' if consenso == 'SUBE' else '#dc2626' if consenso == 'BAJA' else '#f59e0b'

            col_consenso, col_chart = st.columns([1, 2])

            with col_consenso:
                st.markdown(
                    f'<div style="text-align:center; padding:24px; background:#fff; border:1px solid #e2e8f0; '
                    f'border-radius:12px; border-left:4px solid {consenso_color};">'
                    f'<p style="font-size:0.85rem; color:#64748b !important; margin:0;">Consenso de {total} modelos</p>'
                    f'<p style="font-size:2rem; font-weight:800; color:{consenso_color} !important; margin:8px 0;">'
                    f'{consenso_icon} {consenso}</p>'
                    f'<p style="margin:0; font-size:0.9rem;"><b style="color:#16a34a !important;">{sube} Sube</b>'
                    f' &nbsp;|&nbsp; <b style="color:#dc2626 !important;">{baja} Baja</b></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col_chart:
                nombres = list(predicciones_validas.keys())
                confianzas = [v.get('confianza', 0) or 0 for v in predicciones_validas.values()]
                colores = ['#16a34a' if v['prediccion'] == 1 else '#dc2626' for v in predicciones_validas.values()]
                etiquetas = ['Sube' if v['prediccion'] == 1 else 'Baja' for v in predicciones_validas.values()]

                fig = go.Figure(go.Bar(
                    x=nombres,
                    y=[c * 100 for c in confianzas],
                    marker_color=colores,
                    text=[f'{e}<br>{c:.1%}' for e, c in zip(etiquetas, confianzas)],
                    textposition='outside',
                    textfont=dict(size=12, color='#334155'),
                ))
                fig.update_layout(
                    **CHART_LAYOUT,
                    height=300,
                    yaxis=dict(title='Confianza (%)', range=[0, 110], gridcolor='#eef2f6'),
                    xaxis=dict(gridcolor='#eef2f6'),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Tabla resumen ──
        st.markdown('### Detalle por Modelo')
        filas = []
        for nombre, res in resultados.items():
            error = res.get('error')
            modo = res.get('modo', 'real')
            estado = f'Error: {error}' if error else ('Demo (señales técnicas)' if modo == 'demo' else 'Real')
            filas.append({
                'Modelo': nombre,
                'Tipo': MODELOS_CLS[nombre]['tipo'].upper(),
                'Predicción': '—' if error else ('Sube ↑' if res['prediccion'] == 1 else 'Baja ↓'),
                'Confianza': '—' if error else (f'{res["confianza"]:.1%}' if res.get('confianza') else 'N/D'),
                'Modo': estado,
                'Descripción': MODELOS_CLS[nombre]['desc'],
            })
        df_resumen = pd.DataFrame(filas)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)

        # ── Guardar en BD ──
        with st.expander('Guardar predicciones en base de datos'):
            if st.button('Guardar todas las predicciones', key='guardar_cls'):
                guardadas = 0
                for nombre, res in predicciones_validas.items():
                    etiqueta = 'Sube' if res['prediccion'] == 1 else 'Baja'
                    guardar_prediccion(
                        empresa=empresa_key,
                        modelo=nombre,
                        tipo='clasificacion',
                        resultado=etiqueta,
                        confianza=res.get('confianza'),
                        fecha_objetivo=datetime.now().strftime('%Y-%m-%d')
                    )
                    guardadas += 1
                st.success(f'{guardadas} predicciones guardadas en la base de datos.')

        # ── Señal de Trading ──
        st.divider()
        st.markdown('### Señal de Trading')

        confianza_prom = np.mean([v.get('confianza', 0.5) or 0.5 for v in predicciones_validas.values()])
        ratio_sube = sube / total if total > 0 else 0.5

        if ratio_sube >= 0.8 and confianza_prom >= 0.6:
            senal, senal_icon, senal_color, senal_bg = 'COMPRA FUERTE', '🟢', '#16a34a', '#f0fdf4'
            senal_desc = 'La mayoría de modelos predicen alza con alta confianza.'
        elif ratio_sube >= 0.6:
            senal, senal_icon, senal_color, senal_bg = 'COMPRA', '🟢', '#16a34a', '#f0fdf4'
            senal_desc = 'Tendencia alcista probable. Considerar posición larga.'
        elif ratio_sube <= 0.2 and confianza_prom >= 0.6:
            senal, senal_icon, senal_color, senal_bg = 'SHORT', '🔴', '#dc2626', '#fef2f2'
            senal_desc = 'Fuerte señal bajista. Considerar posición corta.'
        elif ratio_sube <= 0.4:
            senal, senal_icon, senal_color, senal_bg = 'VENTA', '🔴', '#dc2626', '#fef2f2'
            senal_desc = 'Tendencia bajista probable. Considerar cerrar posición.'
        else:
            senal, senal_icon, senal_color, senal_bg = 'HOLD', '🟡', '#f59e0b', '#fffbeb'
            senal_desc = 'Señales mixtas. Mantener posición actual y esperar.'

        st.markdown(
            f'<div style="text-align:center; padding:24px; background:{senal_bg}; '
            f'border:1px solid #e2e8f0; border-radius:12px; border-left:4px solid {senal_color};">'
            f'<p style="font-size:0.85rem; color:#64748b !important; margin:0 0 6px 0;">Señal para {empresa_key} — {info["nombre"]}</p>'
            f'<p style="font-size:2.2rem; font-weight:800; color:{senal_color} !important; margin:0;">'
            f'{senal_icon} {senal}</p>'
            f'<p style="font-size:0.9rem; color:#64748b !important; margin:8px 0 0 0;">{senal_desc}</p>'
            f'<p style="font-size:0.78rem; color:#94a3b8 !important; margin:6px 0 0 0;">'
            f'Consenso: {sube}/{total} modelos alcistas | Confianza promedio: {confianza_prom:.1%}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Métricas de Evaluación ──
        st.divider()
        st.markdown('### Métricas de Evaluación de los Modelos')
        metrics_path = os.path.join(MODELS_BASE, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_metrics = json.load(f)
            cls_metrics = all_metrics.get('clasificacion', {}).get(empresa_key, {})

            if cls_metrics:
                met_filas = []
                for nombre_m, met in cls_metrics.items():
                    modo_tag = '' if met.get('modo') == 'real' else ' (Demo)'
                    met_filas.append({
                        'Modelo': f'{nombre_m}{modo_tag}',
                        'Accuracy': f'{met["accuracy"]:.2%}',
                        'Precision': f'{met["precision"]:.2%}',
                        'Recall': f'{met["recall"]:.2%}',
                        'F1-Score': f'{met["f1"]:.2%}',
                        'Muestras Test': met.get('test_samples', '—'),
                    })
                df_met = pd.DataFrame(met_filas)
                st.dataframe(df_met, use_container_width=True, hide_index=True)

                fig_met = go.Figure()
                nombres_m = [m['Modelo'] for m in met_filas]
                for metrica, color in [('accuracy', '#0f2b46'), ('precision', '#1a7f64'), ('recall', '#2563eb'), ('f1', '#9333ea')]:
                    valores = [cls_metrics[n.replace(' (Demo)', '')].get(metrica, 0) * 100
                               for n in [m.split(' (Demo)')[0] for m in nombres_m]]
                    fig_met.add_trace(go.Bar(
                        name=metrica.capitalize(),
                        x=nombres_m, y=valores,
                        marker_color=color, opacity=0.85,
                    ))
                fig_met.update_layout(
                    **CHART_LAYOUT, height=350, barmode='group',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    yaxis=dict(title='Porcentaje (%)', range=[0, 100], gridcolor='#eef2f6'),
                )
                st.plotly_chart(fig_met, use_container_width=True)

                st.caption(
                    '**Accuracy:** % de predicciones correctas | '
                    '**Precision:** De las veces que dijo "Sube", ¿cuántas acertó? | '
                    '**Recall:** De todas las subidas reales, ¿cuántas detectó? | '
                    '**F1-Score:** Balance entre Precision y Recall'
                )
        else:
            st.info('Ejecuta `python -m utils.generate_metrics` para generar las métricas de evaluación.')

    # ── Explicación de modelos ──
    st.divider()
    st.markdown('### Acerca de los Modelos')
    for nombre, cfg in MODELOS_CLS.items():
        with st.expander(f'{nombre}'):
            st.markdown(f'**{cfg["desc"]}**')
            if nombre == 'SVC':
                st.markdown("""
                - **Tipo:** Machine Learning clásico (scikit-learn)
                - **Entrada:** 14 indicadores técnicos del último día
                - **Salida:** Clase binaria (0 = Baja, 1 = Sube)
                - **Optimización:** GridSearchCV sobre kernels RBF/Linear y parámetros C/gamma
                """)
            else:
                st.markdown(f"""
                - **Tipo:** Deep Learning (TensorFlow/Keras)
                - **Entrada:** Secuencia de {VENTANA_DL} días × 14 indicadores
                - **Salida:** Probabilidad (0 a 1), umbral en 0.5
                - **Arquitectura:** {cfg['desc'].split('—')[0].strip()}
                """)
