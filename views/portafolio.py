import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import EMPRESAS
from utils.portfolio_optimizer import (
    obtener_retornos, simulacion_montecarlo,
    portafolio_max_sharpe, portafolio_min_riesgo, recomendar_por_perfil,
)

CHART_LAYOUT = dict(
    template='plotly_white',
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#334155', family='sans-serif', size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    hoverlabel=dict(bgcolor='#0f2b46', font_color='white', font_size=13, bordercolor='#0f2b46'),
)

COLORES_EMPRESAS = ['#0f2b46', '#1a7f64', '#2563eb', '#dc2626', '#f59e0b', '#9333ea']


def render():
    st.markdown('<div class="page-header"><h1>Optimización de Portafolio</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Frontera Eficiente de Markowitz — Encuentra la distribución óptima de tus inversiones</p>',
        unsafe_allow_html=True
    )

    # ── Selección de empresas ──
    col_emp, col_perfil, col_capital = st.columns([3, 1, 1])

    with col_emp:
        empresas_sel = st.multiselect(
            'Empresas a incluir en el portafolio',
            list(EMPRESAS.keys()),
            default=list(EMPRESAS.keys()),
            key='port_empresas'
        )
    with col_perfil:
        perfil = st.selectbox(
            'Perfil de riesgo',
            ['conservador', 'moderado', 'agresivo'],
            index=1, key='port_perfil'
        )
    with col_capital:
        capital = st.number_input('Capital (USD)', min_value=100, value=10000, step=500, key='port_capital')

    if len(empresas_sel) < 2:
        st.warning('Selecciona al menos 2 empresas para optimizar el portafolio.')
        return

    st.divider()

    if st.button('Optimizar portafolio', use_container_width=True, type='primary'):
        tickers = {k: EMPRESAS[k]['ticker'] for k in empresas_sel}

        with st.spinner('Descargando datos históricos (2 años)...'):
            precios, retornos = obtener_retornos(tickers, start='2024-01-01')

        if retornos.empty or len(retornos) < 30:
            st.error('No hay datos suficientes para las empresas seleccionadas.')
            return

        activos = list(retornos.columns)

        with st.spinner('Simulando 5,000 portafolios (Montecarlo)...'):
            df_sim = simulacion_montecarlo(retornos, n_portafolios=5000)

        max_sh = portafolio_max_sharpe(df_sim, activos)
        min_ri = portafolio_min_riesgo(df_sim, activos)
        recomendado, desc_rec = recomendar_por_perfil(max_sh, min_ri, perfil)

        # ── Recomendación ──
        perfil_icons = {'conservador': '🛡️', 'moderado': '⚖️', 'agresivo': '🔥'}
        perfil_colors = {'conservador': '#2563eb', 'moderado': '#f59e0b', 'agresivo': '#dc2626'}

        st.markdown(
            f'<div style="padding:20px; background:#f0fdf4; border:1px solid #bbf7d0; '
            f'border-radius:12px; border-left:4px solid #16a34a; margin-bottom:1rem;">'
            f'<p style="margin:0 0 4px 0; font-weight:700; font-size:1.1rem; color:#166534 !important;">'
            f'{perfil_icons[perfil]} Recomendación para perfil {perfil}: {desc_rec}</p>'
            f'<p style="margin:0; color:#166534 !important; font-size:0.9rem;">'
            f'Retorno esperado: {recomendado["retorno"]:.1%} | '
            f'Riesgo: {recomendado["volatilidad"]:.1%} | '
            f'Sharpe: {recomendado["sharpe"]:.2f}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Distribución recomendada ──
        st.markdown('### Distribución Recomendada')

        col_pie, col_pesos = st.columns([1, 1])

        with col_pie:
            labels = list(recomendado['pesos'].keys())
            values = [v * 100 for v in recomendado['pesos'].values()]
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                marker=dict(colors=COLORES_EMPRESAS[:len(labels)]),
                textinfo='label+percent',
                textfont=dict(size=13, color='white'),
                hole=0.4,
            ))
            fig_pie.update_layout(**CHART_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_pesos:
            st.markdown('#### Asignación por empresa')
            for emp, peso in sorted(recomendado['pesos'].items(), key=lambda x: -x[1]):
                monto = capital * peso
                nombre_full = EMPRESAS[emp]['nombre']
                bar_width = max(peso * 100, 2)
                st.markdown(
                    f'<div style="margin:8px 0;">'
                    f'<div style="display:flex; justify-content:space-between; margin-bottom:2px;">'
                    f'<span style="font-weight:600; font-size:0.9rem;">{emp} <span style="color:#94a3b8 !important; font-weight:400;">({nombre_full})</span></span>'
                    f'<span style="font-weight:700;">{peso:.1%} — ${monto:,.0f}</span>'
                    f'</div>'
                    f'<div style="background:#e2e8f0; border-radius:4px; height:8px; overflow:hidden;">'
                    f'<div style="background:#1a7f64; width:{bar_width}%; height:100%; border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Frontera eficiente ──
        st.markdown('### Frontera Eficiente')
        fig_ef = go.Figure()

        fig_ef.add_trace(go.Scatter(
            x=df_sim['Volatilidad'] * 100,
            y=df_sim['Retorno'] * 100,
            mode='markers',
            marker=dict(
                size=3, color=df_sim['Sharpe'],
                colorscale='Viridis', showscale=True,
                colorbar=dict(title='Sharpe'),
            ),
            name='Portafolios simulados',
            hovertemplate='Retorno: %{y:.1f}%<br>Riesgo: %{x:.1f}%<br>Sharpe: %{marker.color:.2f}',
        ))

        fig_ef.add_trace(go.Scatter(
            x=[max_sh['volatilidad'] * 100],
            y=[max_sh['retorno'] * 100],
            mode='markers+text',
            marker=dict(size=15, color='#dc2626', symbol='star'),
            text=['Max Sharpe'], textposition='top center',
            name=f'Max Sharpe ({max_sh["sharpe"]:.2f})',
        ))

        fig_ef.add_trace(go.Scatter(
            x=[min_ri['volatilidad'] * 100],
            y=[min_ri['retorno'] * 100],
            mode='markers+text',
            marker=dict(size=15, color='#2563eb', symbol='diamond'),
            text=['Min Riesgo'], textposition='top center',
            name=f'Min Riesgo ({min_ri["volatilidad"]:.1%})',
        ))

        fig_ef.update_layout(
            **CHART_LAYOUT, height=450,
            xaxis=dict(title='Riesgo — Volatilidad (%)', gridcolor='#eef2f6'),
            yaxis=dict(title='Retorno Anual (%)', gridcolor='#eef2f6'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # ── Comparativa: Max Sharpe vs Min Riesgo ──
        st.markdown('### Comparación de Portafolios Óptimos')
        col_ms, col_mr = st.columns(2)

        with col_ms:
            st.markdown(
                '<div style="padding:16px; background:#fef2f2; border:1px solid #fecaca; '
                'border-radius:10px; border-top:3px solid #dc2626;">'
                '<p style="font-weight:700; text-align:center; margin:0 0 8px 0;">⭐ Máximo Sharpe Ratio</p>',
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)
            c1.metric('Retorno', f'{max_sh["retorno"]:.1%}')
            c2.metric('Riesgo', f'{max_sh["volatilidad"]:.1%}')
            c3.metric('Sharpe', f'{max_sh["sharpe"]:.2f}')
            for emp, peso in sorted(max_sh['pesos'].items(), key=lambda x: -x[1]):
                st.caption(f'{emp}: {peso:.1%}')
            st.markdown('</div>', unsafe_allow_html=True)

        with col_mr:
            st.markdown(
                '<div style="padding:16px; background:#eff6ff; border:1px solid #bfdbfe; '
                'border-radius:10px; border-top:3px solid #2563eb;">'
                '<p style="font-weight:700; text-align:center; margin:0 0 8px 0;">🛡️ Mínimo Riesgo</p>',
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)
            c1.metric('Retorno', f'{min_ri["retorno"]:.1%}')
            c2.metric('Riesgo', f'{min_ri["volatilidad"]:.1%}')
            c3.metric('Sharpe', f'{min_ri["sharpe"]:.2f}')
            for emp, peso in sorted(min_ri['pesos'].items(), key=lambda x: -x[1]):
                st.caption(f'{emp}: {peso:.1%}')
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # ── Matriz de correlación ──
        st.markdown('### Matriz de Correlación')
        corr = retornos.corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale=['#dc2626', '#ffffff', '#16a34a'],
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(
            'La correlación mide cómo se mueven las acciones juntas. '
            'Valores cercanos a +1 = se mueven igual. Cercanos a -1 = se mueven opuesto. '
            'Para diversificar, busca acciones con baja correlación.'
        )

    # ── Explicación ──
    st.divider()
    st.markdown('### Acerca de la Optimización')
    with st.expander('¿Qué es la Frontera Eficiente de Markowitz?'):
        st.markdown("""
        **Harry Markowitz** (Nobel de Economía 1990) demostró que no basta con elegir las mejores acciones individuales.
        Lo importante es cómo se **combinan** entre sí.

        - **Retorno esperado:** Cuánto esperas ganar al año
        - **Riesgo (Volatilidad):** Cuánto puede variar el precio — más volatilidad = más riesgo
        - **Sharpe Ratio:** Retorno por unidad de riesgo (más alto = mejor)

        La **Frontera Eficiente** es la curva de portafolios que dan el **máximo retorno para cada nivel de riesgo**.
        Cualquier portafolio debajo de esa curva es subóptimo.
        """)

    with st.expander('¿Cómo funciona la simulación?'):
        st.markdown("""
        1. Se generan **5,000 combinaciones aleatorias** de pesos para las empresas seleccionadas
        2. Para cada combinación se calcula retorno, riesgo y Sharpe Ratio
        3. Se identifica el portafolio con **mayor Sharpe** (mejor rendimiento ajustado al riesgo)
        4. Se identifica el portafolio con **menor volatilidad** (más seguro)
        5. Según tu perfil de riesgo, se recomienda uno u otro (o un mix)
        """)

    with st.expander('¿Qué significan los perfiles de riesgo?'):
        st.markdown("""
        | Perfil | Portafolio recomendado | Para quién |
        |--------|----------------------|------------|
        | **Conservador** 🛡️ | Mínimo riesgo | Quienes prefieren proteger su capital |
        | **Moderado** ⚖️ | Balance 50/50 | Equilibrio entre crecimiento y seguridad |
        | **Agresivo** 🔥 | Máximo Sharpe | Quienes buscan maximizar retornos |
        """)
