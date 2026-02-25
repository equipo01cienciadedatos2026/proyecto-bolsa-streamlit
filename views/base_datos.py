import streamlit as st
import pandas as pd
from database.db_utils import (
    crear_usuario, obtener_usuarios, actualizar_usuario, eliminar_usuario,
    crear_portafolio, obtener_portafolios, obtener_detalle_portafolio,
    agregar_activo_portafolio, resumen_portafolio,
    registrar_operacion, obtener_operaciones, resumen_operaciones_usuario,
)
from config import EMPRESAS

PERFILES = ['conservador', 'moderado', 'agresivo']
PERFIL_BADGES = {
    'conservador': ('🛡️', '#dbeafe', '#1e40af'),
    'moderado': ('⚖️', '#fef3c7', '#92400e'),
    'agresivo': ('🔥', '#fee2e2', '#991b1b'),
}


def render():
    st.markdown('<div class="page-header"><h1>Gestión de Base de Datos</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Usuarios, portafolios y operaciones — Base de datos SQLite local</p>', unsafe_allow_html=True)

    tab_usuarios, tab_portafolios, tab_operaciones = st.tabs([
        'Usuarios', 'Portafolios', 'Operaciones'
    ])

    # ── TAB USUARIOS ──
    with tab_usuarios:
        col_form, col_list = st.columns([1, 2])

        with col_form:
            st.markdown('#### Nuevo Usuario')
            with st.form('form_nuevo_usuario', clear_on_submit=True):
                nombre = st.text_input('Nombre completo')
                email = st.text_input('Email')
                perfil = st.selectbox('Perfil de riesgo', PERFILES, index=1)
                capital = st.number_input('Capital inicial (USD)', min_value=100.0, value=10000.0, step=500.0)
                submitted = st.form_submit_button('Crear usuario', use_container_width=True)
                if submitted:
                    if not nombre or not email:
                        st.error('Nombre y email son obligatorios.')
                    else:
                        try:
                            uid = crear_usuario(nombre, email, perfil, capital)
                            st.success(f'Usuario creado con ID {uid}')
                            st.rerun()
                        except Exception as e:
                            st.error(f'Error: {e}')

        with col_list:
            st.markdown('#### Usuarios registrados')
            usuarios = obtener_usuarios()
            if not usuarios:
                st.info('No hay usuarios registrados.')
            else:
                df_u = pd.DataFrame(usuarios)
                df_u = df_u.rename(columns={
                    'id': 'ID', 'nombre': 'Nombre', 'email': 'Email',
                    'perfil_riesgo': 'Perfil', 'capital_inicial': 'Capital',
                    'fecha_registro': 'Registro'
                })
                st.dataframe(df_u[['ID', 'Nombre', 'Email', 'Perfil', 'Capital', 'Registro']],
                             use_container_width=True, hide_index=True)

                with st.expander('Eliminar usuario'):
                    user_del = st.selectbox('Seleccionar usuario', [f"{u['id']} - {u['nombre']}" for u in usuarios], key='del_user')
                    if st.button('Eliminar', type='secondary'):
                        uid = int(user_del.split(' - ')[0])
                        eliminar_usuario(uid)
                        st.success('Usuario eliminado.')
                        st.rerun()

    # ── TAB PORTAFOLIOS ──
    with tab_portafolios:
        usuarios = obtener_usuarios()
        if not usuarios:
            st.warning('Primero crea al menos un usuario.')
            return

        usuario_sel = st.selectbox(
            'Seleccionar usuario',
            [f"{u['id']} - {u['nombre']}" for u in usuarios],
            key='port_user'
        )
        uid = int(usuario_sel.split(' - ')[0])

        col_p1, col_p2 = st.columns([1, 2])

        with col_p1:
            st.markdown('#### Nuevo Portafolio')
            with st.form('form_portafolio', clear_on_submit=True):
                nombre_p = st.text_input('Nombre del portafolio')
                desc_p = st.text_area('Descripción', height=80)
                if st.form_submit_button('Crear portafolio', use_container_width=True):
                    if not nombre_p:
                        st.error('El nombre es obligatorio.')
                    else:
                        crear_portafolio(uid, nombre_p, desc_p)
                        st.success('Portafolio creado.')
                        st.rerun()

        with col_p2:
            st.markdown('#### Portafolios del usuario')
            portafolios = obtener_portafolios(uid)
            if not portafolios:
                st.info('Este usuario no tiene portafolios.')
            else:
                for p in portafolios:
                    with st.expander(f"📁 {p['nombre']}  (ID: {p['id']})"):
                        st.caption(p.get('descripcion', ''))
                        detalle = obtener_detalle_portafolio(p['id'])
                        if detalle:
                            df_d = pd.DataFrame(detalle)
                            df_d = df_d.rename(columns={
                                'empresa': 'Empresa', 'cantidad': 'Cant.',
                                'precio_compra': 'Precio Compra', 'fecha_compra': 'Fecha'
                            })
                            st.dataframe(df_d[['Empresa', 'Cant.', 'Precio Compra', 'Fecha']],
                                         use_container_width=True, hide_index=True)

                            resumen = resumen_portafolio(p['id'])
                            total_inv = sum(r['total_cantidad'] * r['precio_promedio'] for r in resumen)
                            st.metric('Total invertido', f'${total_inv:,.2f}')
                        else:
                            st.info('Portafolio vacío.')

                        st.markdown('##### Agregar activo')
                        c1, c2, c3 = st.columns(3)
                        empresa_add = c1.selectbox('Empresa', list(EMPRESAS.keys()), key=f'emp_{p["id"]}')
                        cant_add = c2.number_input('Cantidad', min_value=0.01, value=10.0, key=f'cant_{p["id"]}')
                        precio_add = c3.number_input('Precio (USD)', min_value=0.01, value=10.0, key=f'prec_{p["id"]}')
                        if st.button('Agregar', key=f'add_{p["id"]}'):
                            agregar_activo_portafolio(p['id'], empresa_add, cant_add, precio_add)
                            registrar_operacion(uid, empresa_add, 'compra', cant_add, precio_add, p['id'], 'Agregado desde gestión')
                            st.success(f'{cant_add} acciones de {empresa_add} agregadas.')
                            st.rerun()

    # ── TAB OPERACIONES ──
    with tab_operaciones:
        usuarios = obtener_usuarios()
        if not usuarios:
            st.warning('Primero crea al menos un usuario.')
            return

        col_o1, col_o2 = st.columns([1, 2])

        with col_o1:
            st.markdown('#### Nueva Operación')
            with st.form('form_operacion', clear_on_submit=True):
                user_op = st.selectbox('Usuario', [f"{u['id']} - {u['nombre']}" for u in usuarios], key='op_user')
                empresa_op = st.selectbox('Empresa', list(EMPRESAS.keys()))
                tipo_op = st.radio('Tipo', ['compra', 'venta'], horizontal=True)
                cant_op = st.number_input('Cantidad', min_value=0.01, value=10.0)
                precio_op = st.number_input('Precio unitario (USD)', min_value=0.01, value=10.0)
                notas_op = st.text_input('Notas (opcional)')
                if st.form_submit_button('Registrar operación', use_container_width=True):
                    uid_op = int(user_op.split(' - ')[0])
                    oid = registrar_operacion(uid_op, empresa_op, tipo_op, cant_op, precio_op, notas=notas_op)
                    st.success(f'Operación #{oid} registrada.')
                    st.rerun()

        with col_o2:
            st.markdown('#### Historial de Operaciones')
            filtro_user = st.selectbox(
                'Filtrar por usuario',
                ['Todos'] + [f"{u['id']} - {u['nombre']}" for u in usuarios],
                key='filtro_op'
            )
            uid_filtro = None if filtro_user == 'Todos' else int(filtro_user.split(' - ')[0])

            ops = obtener_operaciones(usuario_id=uid_filtro, limit=100)
            if not ops:
                st.info('No hay operaciones registradas.')
            else:
                df_ops = pd.DataFrame(ops)
                df_ops = df_ops.rename(columns={
                    'id': 'ID', 'usuario_id': 'User', 'empresa': 'Empresa',
                    'tipo': 'Tipo', 'cantidad': 'Cant.', 'precio': 'Precio',
                    'total': 'Total', 'fecha': 'Fecha', 'notas': 'Notas'
                })
                st.dataframe(
                    df_ops[['ID', 'User', 'Empresa', 'Tipo', 'Cant.', 'Precio', 'Total', 'Fecha', 'Notas']],
                    use_container_width=True, hide_index=True
                )

                if uid_filtro:
                    resumen = resumen_operaciones_usuario(uid_filtro)
                    c1, c2, c3 = st.columns(3)
                    c1.metric('Total operaciones', resumen.get('total_ops', 0))
                    c2.metric('Total compras', f"${resumen.get('total_compras', 0):,.2f}")
                    c3.metric('Total ventas', f"${resumen.get('total_ventas', 0):,.2f}")
