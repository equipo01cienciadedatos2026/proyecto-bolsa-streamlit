import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from config import EMPRESAS
from utils.data_loader import obtener_datos_historicos
from utils.backtesting_utils import (
    estrategia_buy_hold,
    estrategia_sma_crossover,
    estrategia_rsi,
    estrategia_macd,
    estrategia_modelo_svc,
)

CHART_LAYOUT = dict(
    template='plotly_white',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#334155', family='sans-serif', size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    hoverlabel=dict(bgcolor='#0f2b46', font_color='white', font_size=13, bordercolor='#0f2b46'),
)

ESTRATEGIAS = {
    'Buy & Hold': {
        'fn': 'buy_hold',
        'desc': 'Comprar al inicio y mantener hasta el final. Es el benchmark de referencia.',
        'params': [],
    },
    'Cruce de SMA': {
        'fn': 'sma',
        'desc': 'Compra cuando la SMA corta cruza por encima de la SMA larga. Vende al cruce inverso.',
        'params': ['sma_corta', 'sma_larga'],
    },
    'RSI': {
        'fn': 'rsi',
        'desc': 'Compra cuando RSI cae por debajo del nivel de sobreventa. Vende cuando supera sobrecompra.',
        'params': ['rsi_sobreventa', 'rsi_sobrecompra'],
    },
    'MACD': {
        'fn': 'macd',
        'desc': 'Compra cuando MACD cruza por encima de la señal. Vende al cruce inverso.',
        'params': [],
    },
    'Modelo SVC': {
        'fn': 'svc',
        'desc': 'Usa las predicciones del modelo SVC entrenado: compra si predice "Sube", vende si predice "Baja".',
        'params': [],
    },
}

COLORES = {
    'Buy & Hold': '#94a3b8',
    'Cruce de SMA': '#2563eb',
    'RSI': '#16a34a',
    'MACD': '#9333ea',
    'Modelo SVC': '#dc2626',
}


def render():
    st.markdown('<div class="page-header"><h1>Backtesting de Estrategias</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Simula estrategias de trading sobre datos históricos — Mide rendimiento antes de invertir</p>',
        unsafe_allow_html=True
    )

    # ── Configuración ──
    col_emp, col_ini, col_fin, col_cap = st.columns([2.5, 1, 1, 1])
    with col_emp:
        opciones = [f'{k} — {v["nombre"]}' for k, v in EMPRESAS.items()]
        seleccion = st.selectbox('EMPRESA', opciones, key='bt_empresa')
        empresa_key = seleccion.split(' — ')[0]
    with col_ini:
        fecha_inicio = st.date_input('DESDE', value=date(2020, 1, 1), key='bt_inicio')
    with col_fin:
        fecha_fin = st.date_input('HASTA', value=date.today(), key='bt_fin')
    with col_cap:
        capital = st.number_input('CAPITAL (USD)', min_value=100, value=10000, step=500, key='bt_capital')

    st.divider()

    # ── Selección de estrategias ──
    st.markdown('#### Estrategias a comparar')

    col_strat, col_params = st.columns([2, 3])

    with col_strat:
        estrategias_sel = st.multiselect(
            'Selecciona estrategias',
            list(ESTRATEGIAS.keys()),
            default=['Buy & Hold', 'Cruce de SMA', 'Modelo SVC'],
            key='bt_estrategias'
        )

    with col_params:
        params = {}
        if 'Cruce de SMA' in estrategias_sel:
            c1, c2 = st.columns(2)
            params['sma_corta'] = c1.number_input('SMA Corta', min_value=5, max_value=50, value=20, key='bt_sma_c')
            params['sma_larga'] = c2.number_input('SMA Larga', min_value=20, max_value=200, value=50, key='bt_sma_l')
        if 'RSI' in estrategias_sel:
            c1, c2 = st.columns(2)
            params['rsi_sobreventa'] = c1.number_input('RSI Sobreventa', min_value=10, max_value=40, value=30, key='bt_rsi_sv')
            params['rsi_sobrecompra'] = c2.number_input('RSI Sobrecompra', min_value=60, max_value=90, value=70, key='bt_rsi_sc')

    if not estrategias_sel:
        st.warning('Selecciona al menos una estrategia.')
        return

    st.divider()

    # ── Ejecutar backtesting ──
    if st.button('Ejecutar backtesting', use_container_width=True, type='primary'):
        info = EMPRESAS[empresa_key]

        with st.spinner('Descargando datos históricos...'):
            df = obtener_datos_historicos(info['ticker'], str(fecha_inicio), str(fecha_fin))
            if df.empty or len(df) < 60:
                st.error('No hay datos suficientes para el período seleccionado (mínimo 60 días).')
                return

        st.markdown(
            f'<p style="color:#64748b; font-size:0.8rem; text-align:right;">'
            f'Período: {len(df)} días de trading | {fecha_inicio.strftime("%d/%m/%Y")} → {fecha_fin.strftime("%d/%m/%Y")}</p>',
            unsafe_allow_html=True
        )

        resultados = {}

        with st.spinner('Ejecutando estrategias...'):
            progress = st.progress(0)

            for idx, nombre in enumerate(estrategias_sel):
                cfg = ESTRATEGIAS[nombre]
                try:
                    if cfg['fn'] == 'buy_hold':
                        equity, trades, metrics = estrategia_buy_hold(df, float(capital))
                    elif cfg['fn'] == 'sma':
                        equity, trades, metrics = estrategia_sma_crossover(
                            df, params.get('sma_corta', 20), params.get('sma_larga', 50), float(capital)
                        )
                    elif cfg['fn'] == 'rsi':
                        equity, trades, metrics = estrategia_rsi(
                            df, sobreventa=params.get('rsi_sobreventa', 30),
                            sobrecompra=params.get('rsi_sobrecompra', 70), capital=float(capital)
                        )
                    elif cfg['fn'] == 'macd':
                        equity, trades, metrics = estrategia_macd(df, capital=float(capital))
                    elif cfg['fn'] == 'svc':
                        equity, trades, metrics = estrategia_modelo_svc(df, empresa_key, float(capital))
                        if equity is None:
                            resultados[nombre] = {'error': 'Modelo SVC no disponible para esta empresa'}
                            progress.progress((idx + 1) / len(estrategias_sel))
                            continue

                    resultados[nombre] = {'equity': equity, 'trades': trades, 'metrics': metrics}
                except Exception as e:
                    resultados[nombre] = {'error': str(e)}

                progress.progress((idx + 1) / len(estrategias_sel))
            progress.empty()

        validos = {k: v for k, v in resultados.items() if 'metrics' in v}
        errores = {k: v for k, v in resultados.items() if 'error' in v}

        if errores:
            for nombre, res in errores.items():
                st.warning(f'**{nombre}:** {res["error"]}')

        if not validos:
            st.error('Ninguna estrategia se ejecutó correctamente.')
            return

        # ── Métricas principales ──
        st.markdown('### Resultados')
        cols_met = st.columns(len(validos))
        for i, (nombre, res) in enumerate(validos.items()):
            m = res['metrics']
            with cols_met[i]:
                color = '#16a34a' if m['retorno_total'] >= 0 else '#dc2626'
                bg = '#f0fdf4' if m['retorno_total'] >= 0 else '#fef2f2'
                st.markdown(
                    f'<div style="text-align:center; padding:18px 12px; background:{bg}; '
                    f'border:1px solid #e2e8f0; border-radius:10px; border-top:3px solid {COLORES.get(nombre, "#64748b")};">'
                    f'<p style="font-weight:700; font-size:0.9rem; color:#64748b !important; margin:0 0 8px 0;">{nombre}</p>'
                    f'<p style="font-size:1.4rem; font-weight:800; color:{color} !important; margin:0;">'
                    f'{m["retorno_total"]:+.2f}%</p>'
                    f'<p style="font-size:0.8rem; color:#64748b !important; margin:4px 0 0 0;">'
                    f'${m["capital_final"]:,.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Curva de equity ──
        st.markdown('### Curva de Equity')
        fig = go.Figure()
        for nombre, res in validos.items():
            equity = res['equity']
            fechas_eq = pd.to_datetime(df['Date']).iloc[:len(equity)]
            dash = 'dash' if nombre == 'Buy & Hold' else 'solid'
            fig.add_trace(go.Scatter(
                x=fechas_eq, y=equity,
                mode='lines', name=nombre,
                line=dict(color=COLORES.get(nombre, '#64748b'), width=2, dash=dash),
            ))

        fig.add_hline(
            y=float(capital), line_dash='dot', line_color='#cbd5e1',
            annotation_text=f'Capital inicial: ${capital:,}',
            annotation_font_color='#94a3b8',
        )
        fig.update_layout(
            **CHART_LAYOUT, height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis=dict(title='Valor del Portafolio (USD)', gridcolor='#eef2f6'),
            xaxis=dict(gridcolor='#eef2f6'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Tabla comparativa ──
        st.markdown('### Comparación de Métricas')
        filas = []
        for nombre, res in validos.items():
            m = res['metrics']
            filas.append({
                'Estrategia': nombre,
                'Retorno Total': f'{m["retorno_total"]:+.2f}%',
                'Sharpe Ratio': f'{m["sharpe_ratio"]:.2f}',
                'Max Drawdown': f'{m["max_drawdown"]:.2f}%',
                'Win Rate': f'{m["win_rate"]:.1f}%',
                'Total Trades': m['total_trades'],
                'P&L Total': f'${m["pnl_total"]:+,.2f}',
                'Capital Final': f'${m["capital_final"]:,.2f}',
            })
        df_comp = pd.DataFrame(filas)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # ── Mejor estrategia ──
        mejor = max(validos.items(), key=lambda x: x[1]['metrics']['retorno_total'])
        mejor_nombre = mejor[0]
        mejor_ret = mejor[1]['metrics']['retorno_total']
        mejor_sharpe = mejor[1]['metrics']['sharpe_ratio']

        st.markdown(
            f'<div style="padding:16px 20px; background:#f0fdf4; border:1px solid #bbf7d0; '
            f'border-radius:10px; border-left:4px solid #16a34a; margin:1rem 0;">'
            f'<p style="margin:0; font-weight:700; color:#166534 !important;">Mejor estrategia: {mejor_nombre}</p>'
            f'<p style="margin:4px 0 0 0; color:#166534 !important; font-size:0.9rem;">'
            f'Retorno: {mejor_ret:+.2f}% | Sharpe: {mejor_sharpe:.2f}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Historial de trades ──
        with st.expander('Historial de operaciones por estrategia'):
            tab_names = list(validos.keys())
            tabs = st.tabs(tab_names)
            for tab, nombre in zip(tabs, tab_names):
                with tab:
                    trades = validos[nombre]['trades']
                    if not trades:
                        st.info('Sin operaciones.')
                        continue
                    trade_rows = []
                    for t in trades:
                        trade_rows.append({
                            'Tipo': t['tipo'].capitalize(),
                            'Fecha': pd.to_datetime(t['fecha']).strftime('%d/%m/%Y') if not isinstance(t['fecha'], str) else t['fecha'],
                            'Precio': f'${t["precio"]:.2f}',
                            'Cantidad': f'{t["cantidad"]:.2f}',
                            'P&L': f'${t["pnl"]:+.2f}' if t.get('pnl') is not None else '—',
                        })
                    st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

    # ── Explicación ──
    st.divider()
    st.markdown('### Acerca de las Estrategias')
    for nombre, cfg in ESTRATEGIAS.items():
        with st.expander(nombre):
            st.markdown(f'**{cfg["desc"]}**')
            if nombre == 'Buy & Hold':
                st.markdown("""
                - **Tipo:** Benchmark pasivo
                - **Lógica:** Comprar todo al inicio, no hacer nada más
                - **Uso:** Sirve como referencia — si tu estrategia no supera Buy & Hold, no vale la pena
                """)
            elif nombre == 'Cruce de SMA':
                st.markdown("""
                - **Tipo:** Seguimiento de tendencia
                - **Señal de compra:** SMA corta (ej: 20 días) cruza POR ENCIMA de SMA larga (ej: 50 días)
                - **Señal de venta:** SMA corta cruza POR DEBAJO de SMA larga
                - **Ventaja:** Captura tendencias fuertes. **Debilidad:** Señales falsas en mercados laterales
                """)
            elif nombre == 'RSI':
                st.markdown("""
                - **Tipo:** Oscilador / Reversión a la media
                - **Señal de compra:** RSI < 30 (sobreventa — el precio cayó demasiado)
                - **Señal de venta:** RSI > 70 (sobrecompra — el precio subió demasiado)
                - **Ventaja:** Bueno en mercados laterales. **Debilidad:** Puede vender muy pronto en tendencias fuertes
                """)
            elif nombre == 'MACD':
                st.markdown("""
                - **Tipo:** Momentum / Seguimiento de tendencia
                - **Señal de compra:** MACD cruza POR ENCIMA de su línea de señal
                - **Señal de venta:** MACD cruza POR DEBAJO de su línea de señal
                - **Ventaja:** Combina tendencia y momentum. **Debilidad:** Retraso en señales
                """)
            elif nombre == 'Modelo SVC':
                st.markdown("""
                - **Tipo:** Machine Learning
                - **Señal de compra:** El modelo SVC predice "Sube" (clase 1)
                - **Señal de venta:** El modelo SVC predice "Baja" (clase 0)
                - **Ventaja:** Usa 14 indicadores técnicos simultáneamente
                - **Debilidad:** Depende de la calidad del entrenamiento
                """)
