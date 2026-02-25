import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime
from config import EMPRESAS
from utils.data_loader import obtener_precios_actuales, obtener_datos_historicos


CHART_LAYOUT = dict(
    template='plotly_white',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#334155', family='sans-serif', size=12),
    xaxis=dict(gridcolor='#eef2f6', linecolor='#e2e8f0', zeroline=False),
    yaxis=dict(gridcolor='#eef2f6', linecolor='#e2e8f0', zeroline=False),
    margin=dict(l=0, r=0, t=10, b=0),
    hoverlabel=dict(bgcolor='#0f2b46', font_color='white', font_size=13, bordercolor='#0f2b46'),
)


def render():
    st.markdown('<div class="page-header"><h1>Dashboard de Mercado</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Vista general de indicadores financieros — Empresas mineras con operaciones en Perú</p>', unsafe_allow_html=True)

    now = datetime.now()
    hora_str = now.strftime('%H:%M')
    fecha_str = now.strftime('%d/%m/%Y')
    st.markdown(
        f'<p style="text-align:right; color:#94a3b8 !important; font-size:0.78rem; margin-top:-10px;">'
        f'Última actualización: {fecha_str} a las {hora_str}</p>',
        unsafe_allow_html=True
    )

    precios = obtener_precios_actuales()

    cols = st.columns(6)
    for i, (key, info) in enumerate(EMPRESAS.items()):
        with cols[i]:
            datos = precios.get(key)
            if datos:
                cambio = datos['cambio']
                badge_cls = 'badge-green' if cambio >= 0 else 'badge-red'
                arrow = '▲' if cambio >= 0 else '▼'
                st.metric(
                    label=key,
                    value=f'${datos["precio"]:.2f}',
                    delta=f'{cambio:+.2f}%'
                )
                st.caption(info['nombre'])
            else:
                st.metric(label=key, value='—')
                st.caption(info['nombre'])

    st.divider()

    col_sel, col_periodo, col_ini, col_fin = st.columns([2.5, 1, 1, 1])
    with col_sel:
        opciones = [f'{k} — {v["nombre"]}' for k, v in EMPRESAS.items()]
        seleccion = st.selectbox('EMPRESA', opciones)
        empresa_key = seleccion.split(' — ')[0]
    with col_periodo:
        periodo = st.selectbox('PERÍODO', ['Personalizado', '1 Mes', '3 Meses', '6 Meses', '1 Año', '5 Años'])
    with col_ini:
        if periodo == 'Personalizado':
            fecha_inicio = st.date_input('DESDE', value=date(2020, 1, 1))
        else:
            meses = {'1 Mes': 30, '3 Meses': 90, '6 Meses': 180, '1 Año': 365, '5 Años': 1825}
            dias = meses[periodo]
            fecha_inicio = date.today() - pd.Timedelta(days=dias)
            st.date_input('DESDE', value=fecha_inicio, disabled=True)
    with col_fin:
        fecha_fin = st.date_input('HASTA', value=date.today())

    info = EMPRESAS[empresa_key]
    df = obtener_datos_historicos(info['ticker'], str(fecha_inicio), str(fecha_fin))

    if df.empty:
        st.warning('No se encontraron datos para el rango seleccionado.')
        return

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin:1rem 0 0.5rem 0;">
        <h2 style="margin:0 !important;">{info['nombre']}</h2>
        <span class="badge badge-blue">{info['bolsa']}</span>
        <span class="badge badge-blue">{empresa_key}</span>
    </div>
    """, unsafe_allow_html=True)

    ultimo = df.iloc[-1]
    primero = df.iloc[0]
    retorno_total = ((float(ultimo['Close']) / float(primero['Close'])) - 1) * 100
    precio_max = float(df['High'].max())
    precio_min = float(df['Low'].min())
    vol_prom = df['Volume'].mean()

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric('Precio Actual', f'${float(ultimo["Close"]):.2f}')
    col_m2.metric('Retorno Período', f'{retorno_total:+.2f}%')
    col_m3.metric('Máximo', f'${precio_max:.2f}')
    col_m4.metric('Mínimo', f'${precio_min:.2f}')
    col_m5.metric('Vol. Promedio', f'{vol_prom:,.0f}')

    tab_velas, tab_linea, tab_volumen, tab_datos = st.tabs(
        ['Velas Japonesas', 'Línea + Medias Móviles', 'Volumen', 'Datos OHLCV']
    )

    with tab_velas:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            increasing_line_color='#16a34a', increasing_fillcolor='#bbf7d0',
            decreasing_line_color='#dc2626', decreasing_fillcolor='#fecaca',
            name='OHLC'
        ), row=1, col=1)
        vol_colors = ['#16a34a' if c >= o else '#dc2626'
                      for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            marker_color=vol_colors, opacity=0.5, name='Volumen',
            showlegend=False
        ), row=2, col=1)
        fig.update_layout(**CHART_LAYOUT, height=520,
                          xaxis_rangeslider_visible=False, showlegend=False)
        fig.update_yaxes(title_text='Precio (USD)', row=1, col=1)
        fig.update_yaxes(title_text='Vol.', row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab_linea:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name='Cierre',
            line=dict(color='#0f2b46', width=2.2)
        ))
        if len(df) >= 20:
            sma20 = df['Close'].rolling(20).mean()
            fig2.add_trace(go.Scatter(
                x=df['Date'], y=sma20,
                mode='lines', name='SMA 20',
                line=dict(color='#f59e0b', width=1.5, dash='dot')
            ))
        if len(df) >= 50:
            sma50 = df['Close'].rolling(50).mean()
            fig2.add_trace(go.Scatter(
                x=df['Date'], y=sma50,
                mode='lines', name='SMA 50',
                line=dict(color='#dc2626', width=1.5, dash='dash')
            ))
        fig2.update_layout(**CHART_LAYOUT, height=420,
                           legend=dict(orientation='h', yanchor='bottom',
                                       y=1.02, xanchor='right', x=1))
        fig2.update_yaxes(title_text='Precio (USD)')
        st.plotly_chart(fig2, use_container_width=True)

    with tab_volumen:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            marker_color='#1a7f64', opacity=0.65, name='Volumen'
        ))
        fig3.update_layout(**CHART_LAYOUT, height=300)
        fig3.update_yaxes(title_text='Volumen')
        st.plotly_chart(fig3, use_container_width=True)

    with tab_datos:
        st.dataframe(
            df.style.format({
                'Open': '${:.2f}', 'High': '${:.2f}',
                'Low': '${:.2f}', 'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True, height=400
        )

    st.divider()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🔮</div>
            <h3>Modelos de Clasificación</h3>
            <p>Predicciones de tendencia (Sube/Baja) para el próximo día con 5 modelos: SVC, SimpleRNN, LSTM, BiLSTM y GRU.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">📊</div>
            <h3>Modelos de Regresión</h3>
            <p>Predicciones de precio con ARIMA, LSTM Regressor y el ensamblaje ARIMA-LSTM para mayor precisión.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">📈</div>
            <h3>Backtesting</h3>
            <p>Simulación de estrategias de trading con métricas: retorno, Sharpe Ratio, Max Drawdown y más.</p>
        </div>
        """, unsafe_allow_html=True)
