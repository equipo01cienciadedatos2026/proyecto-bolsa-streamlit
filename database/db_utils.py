"""
Funciones CRUD para la base de datos SQLite de FinPredict.
"""
import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'bolsa.db')


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Usuarios ──

def crear_usuario(nombre, email, perfil_riesgo='moderado', capital_inicial=10000.0):
    with get_conn() as conn:
        conn.execute(
            'INSERT INTO usuarios (nombre, email, perfil_riesgo, capital_inicial) VALUES (?,?,?,?)',
            (nombre, email, perfil_riesgo, capital_inicial)
        )
        return conn.execute('SELECT last_insert_rowid()').fetchone()[0]


def obtener_usuarios():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM usuarios ORDER BY nombre').fetchall()]


def obtener_usuario(usuario_id):
    with get_conn() as conn:
        row = conn.execute('SELECT * FROM usuarios WHERE id = ?', (usuario_id,)).fetchone()
        return dict(row) if row else None


def actualizar_usuario(usuario_id, nombre, email, perfil_riesgo, capital_inicial):
    with get_conn() as conn:
        conn.execute(
            'UPDATE usuarios SET nombre=?, email=?, perfil_riesgo=?, capital_inicial=? WHERE id=?',
            (nombre, email, perfil_riesgo, capital_inicial, usuario_id)
        )


def eliminar_usuario(usuario_id):
    with get_conn() as conn:
        conn.execute('DELETE FROM operaciones WHERE usuario_id = ?', (usuario_id,))
        conn.execute('DELETE FROM portafolio_detalle WHERE portafolio_id IN (SELECT id FROM portafolios WHERE usuario_id = ?)', (usuario_id,))
        conn.execute('DELETE FROM portafolios WHERE usuario_id = ?', (usuario_id,))
        conn.execute('DELETE FROM usuarios WHERE id = ?', (usuario_id,))


# ── Portafolios ──

def crear_portafolio(usuario_id, nombre, descripcion=''):
    with get_conn() as conn:
        conn.execute(
            'INSERT INTO portafolios (usuario_id, nombre, descripcion) VALUES (?,?,?)',
            (usuario_id, nombre, descripcion)
        )
        return conn.execute('SELECT last_insert_rowid()').fetchone()[0]


def obtener_portafolios(usuario_id):
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            'SELECT * FROM portafolios WHERE usuario_id = ? AND activo = 1 ORDER BY fecha_creacion DESC',
            (usuario_id,)
        ).fetchall()]


def obtener_detalle_portafolio(portafolio_id):
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            'SELECT * FROM portafolio_detalle WHERE portafolio_id = ? ORDER BY fecha_compra DESC',
            (portafolio_id,)
        ).fetchall()]


def agregar_activo_portafolio(portafolio_id, empresa, cantidad, precio_compra):
    with get_conn() as conn:
        conn.execute(
            'INSERT INTO portafolio_detalle (portafolio_id, empresa, cantidad, precio_compra) VALUES (?,?,?,?)',
            (portafolio_id, empresa, cantidad, precio_compra)
        )


# ── Operaciones ──

def registrar_operacion(usuario_id, empresa, tipo, cantidad, precio, portafolio_id=None, notas=''):
    with get_conn() as conn:
        conn.execute(
            'INSERT INTO operaciones (usuario_id, portafolio_id, empresa, tipo, cantidad, precio, notas) VALUES (?,?,?,?,?,?,?)',
            (usuario_id, portafolio_id, empresa, tipo, cantidad, precio, notas)
        )
        return conn.execute('SELECT last_insert_rowid()').fetchone()[0]


def obtener_operaciones(usuario_id=None, empresa=None, limit=50):
    with get_conn() as conn:
        query = 'SELECT * FROM operaciones WHERE 1=1'
        params = []
        if usuario_id:
            query += ' AND usuario_id = ?'
            params.append(usuario_id)
        if empresa:
            query += ' AND empresa = ?'
            params.append(empresa)
        query += ' ORDER BY fecha DESC LIMIT ?'
        params.append(limit)
        return [dict(r) for r in conn.execute(query, params).fetchall()]


# ── Predicciones ──

def guardar_prediccion(empresa, modelo, tipo, resultado, confianza=None, fecha_objetivo=None):
    with get_conn() as conn:
        conn.execute(
            'INSERT INTO predicciones (empresa, modelo, tipo, resultado, confianza, fecha_objetivo) VALUES (?,?,?,?,?,?)',
            (empresa, modelo, tipo, resultado, confianza, fecha_objetivo)
        )


def obtener_predicciones(empresa=None, tipo=None, limit=20):
    with get_conn() as conn:
        query = 'SELECT * FROM predicciones WHERE 1=1'
        params = []
        if empresa:
            query += ' AND empresa = ?'
            params.append(empresa)
        if tipo:
            query += ' AND tipo = ?'
            params.append(tipo)
        query += ' ORDER BY fecha_prediccion DESC LIMIT ?'
        params.append(limit)
        return [dict(r) for r in conn.execute(query, params).fetchall()]


# ── Estadísticas ──

def resumen_portafolio(portafolio_id):
    """Retorna un resumen con total invertido y posiciones."""
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT empresa, SUM(cantidad) as total_cantidad, AVG(precio_compra) as precio_promedio '
            'FROM portafolio_detalle WHERE portafolio_id = ? GROUP BY empresa',
            (portafolio_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def resumen_operaciones_usuario(usuario_id):
    """Retorna totales de compras y ventas de un usuario."""
    with get_conn() as conn:
        row = conn.execute(
            'SELECT '
            'COUNT(*) as total_ops, '
            'SUM(CASE WHEN tipo="compra" THEN total ELSE 0 END) as total_compras, '
            'SUM(CASE WHEN tipo="venta" THEN total ELSE 0 END) as total_ventas '
            'FROM operaciones WHERE usuario_id = ?',
            (usuario_id,)
        ).fetchone()
        return dict(row) if row else {}
