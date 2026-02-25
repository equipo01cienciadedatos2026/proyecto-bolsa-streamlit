"""
Datos de muestra para la base de datos de FinPredict.
Ejecutar después de init_db.py.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'bolsa.db')


def seed():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    usuarios_count = c.execute('SELECT COUNT(*) FROM usuarios').fetchone()[0]
    if usuarios_count > 0:
        print('La base de datos ya contiene datos. Seed omitido.')
        conn.close()
        return

    usuarios = [
        ('Carlos Mendoza', 'carlos.mendoza@unmsm.edu.pe', 'moderado', 15000.0),
        ('María López', 'maria.lopez@unmsm.edu.pe', 'agresivo', 20000.0),
        ('Juan Pérez', 'juan.perez@unmsm.edu.pe', 'conservador', 10000.0),
    ]
    c.executemany(
        'INSERT INTO usuarios (nombre, email, perfil_riesgo, capital_inicial) VALUES (?,?,?,?)',
        usuarios
    )

    portafolios = [
        (1, 'Portafolio Minero', 'Enfocado en empresas mineras con operaciones en Perú'),
        (2, 'Portafolio Agresivo', 'Orientado a alto rendimiento en minería de cobre y oro'),
        (3, 'Portafolio Conservador', 'Empresas grandes y estables del sector'),
    ]
    c.executemany(
        'INSERT INTO portafolios (usuario_id, nombre, descripcion) VALUES (?,?,?)',
        portafolios
    )

    detalle = [
        (1, 'FSM', 50, 4.25),
        (1, 'BVN', 30, 12.50),
        (1, 'SCCO', 10, 85.30),
        (2, 'ABX', 40, 18.90),
        (2, 'BHP', 15, 62.40),
        (2, 'FSM', 100, 3.80),
        (3, 'SCCO', 20, 88.10),
        (3, 'BHP', 25, 60.50),
    ]
    c.executemany(
        'INSERT INTO portafolio_detalle (portafolio_id, empresa, cantidad, precio_compra) VALUES (?,?,?,?)',
        detalle
    )

    operaciones = [
        (1, 1, 'FSM', 'compra', 50, 4.25, 'Compra inicial'),
        (1, 1, 'BVN', 'compra', 30, 12.50, 'Compra inicial'),
        (1, 1, 'SCCO', 'compra', 10, 85.30, 'Compra inicial'),
        (2, 2, 'ABX', 'compra', 40, 18.90, 'Posición en oro'),
        (2, 2, 'BHP', 'compra', 15, 62.40, 'Diversificación'),
        (2, 2, 'FSM', 'compra', 100, 3.80, 'Apuesta agresiva plata'),
        (3, 3, 'SCCO', 'compra', 20, 88.10, 'Compra conservadora'),
        (3, 3, 'BHP', 'compra', 25, 60.50, 'Empresa estable'),
        (1, 1, 'FSM', 'venta', 10, 5.10, 'Toma de ganancias parcial'),
        (2, 2, 'ABX', 'venta', 10, 20.50, 'Venta parcial oro'),
    ]
    c.executemany(
        'INSERT INTO operaciones (usuario_id, portafolio_id, empresa, tipo, cantidad, precio, notas) VALUES (?,?,?,?,?,?,?)',
        operaciones
    )

    conn.commit()
    conn.close()
    print('Datos de muestra insertados correctamente.')


if __name__ == '__main__':
    seed()
