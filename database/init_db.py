"""
Creación del esquema de base de datos SQLite para FinPredict.
Tablas: usuarios, portafolios, portafolio_detalle, operaciones, predicciones.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'bolsa.db')


def crear_tablas():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        perfil_riesgo TEXT DEFAULT 'moderado' CHECK(perfil_riesgo IN ('conservador','moderado','agresivo')),
        capital_inicial REAL DEFAULT 10000.0,
        fecha_registro TEXT DEFAULT (datetime('now','localtime'))
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS portafolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario_id INTEGER NOT NULL,
        nombre TEXT NOT NULL,
        descripcion TEXT,
        fecha_creacion TEXT DEFAULT (datetime('now','localtime')),
        activo INTEGER DEFAULT 1,
        FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS portafolio_detalle (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portafolio_id INTEGER NOT NULL,
        empresa TEXT NOT NULL,
        cantidad REAL NOT NULL DEFAULT 0,
        precio_compra REAL NOT NULL,
        fecha_compra TEXT DEFAULT (datetime('now','localtime')),
        FOREIGN KEY (portafolio_id) REFERENCES portafolios(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS operaciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario_id INTEGER NOT NULL,
        portafolio_id INTEGER,
        empresa TEXT NOT NULL,
        tipo TEXT NOT NULL CHECK(tipo IN ('compra','venta')),
        cantidad REAL NOT NULL,
        precio REAL NOT NULL,
        total REAL GENERATED ALWAYS AS (cantidad * precio) STORED,
        fecha TEXT DEFAULT (datetime('now','localtime')),
        notas TEXT,
        FOREIGN KEY (usuario_id) REFERENCES usuarios(id),
        FOREIGN KEY (portafolio_id) REFERENCES portafolios(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS predicciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        empresa TEXT NOT NULL,
        modelo TEXT NOT NULL,
        tipo TEXT NOT NULL CHECK(tipo IN ('clasificacion','regresion')),
        resultado TEXT NOT NULL,
        confianza REAL,
        fecha_prediccion TEXT DEFAULT (datetime('now','localtime')),
        fecha_objetivo TEXT
    )''')

    conn.commit()
    conn.close()
    print(f'Base de datos creada en: {DB_PATH}')


if __name__ == '__main__':
    crear_tablas()
