# FinPredict — Sistema de Prediccion Bursatil con IA

Sistema de inteligencia de negocios para prediccion de tendencias y precios de acciones mineras, desarrollado con **Streamlit**, **scikit-learn**, **TensorFlow** y **Yahoo Finance**.

**UNMSM — Inteligencia de Negocios | Semestre 2026-0**

## Grupo 1

### Integrantes

* Asencios Rojas, Herberth Alvaro.
* Benites Meza, Marco Fabricio.
* Del Solar Rojas, Jorge Sebastian.
* Guerrero Jaramillo, Andres Abraham.
* Peralta Farfan, Raymond Alain.
* Matos Ramos, Franco Antonio.
* Herrera Fernandez Yumerth Mijail.
* Vidalon Flores, Daniel Omar.
* Rojas Humani, Percy Ares.

## Empresas Analizadas

| Clave | Empresa | Bolsa | Pais |
|-------|---------|-------|------|
| FSM | Fortuna Silver Mines | NYSE | Canada |
| VOLCABC1 | Volcan Compania Minera | BVL | Peru |
| BVN | Buenaventura | NYSE | Peru |
| ABX | Barrick Gold | TSX | Canada |
| BHP | BHP Billiton | NYSE | Australia |
| SCCO | Southern Copper | NYSE | USA |

## Modulos del Sistema

### 1. Dashboard de Mercado
Datos en tiempo real de Yahoo Finance: precios, variaciones, graficos de velas japonesas, lineas con medias moviles (SMA 20/50) y volumen.

### 2. Modelos de Clasificacion (Tendencia)
Prediccion de si la accion **sube o baja** al dia siguiente con 5 modelos:
- **SVC** — Support Vector Classifier (scikit-learn)
- **SimpleRNN** — Red neuronal recurrente basica (Keras)
- **LSTM** — Long Short-Term Memory (Keras)
- **BiLSTM** — Bidirectional LSTM (Keras)
- **GRU** — Gated Recurrent Unit (Keras)

Incluye consenso por votacion y grafico comparativo de confianza.

### 3. Modelos de Regresion (Precio)
Prediccion del **precio exacto** para los proximos N dias con 3 modelos:
- **ARIMA** — Modelo estadistico clasico (statsmodels/pmdarima)
- **LSTM Regressor** — Red neuronal para series temporales (Keras)
- **ARIMA-LSTM** — Ensamblaje que promedia ambos modelos

Incluye grafico de proyeccion y tabla dia a dia.

### 4. Backtesting
Simulacion de estrategias de trading sobre datos historicos:
- **Buy & Hold** — Benchmark pasivo
- **Cruce de SMA** — Seguimiento de tendencia
- **RSI** — Oscilador de sobrecompra/sobreventa
- **MACD** — Momentum y tendencia
- **Modelo SVC** — Estrategia basada en IA

Metricas: Retorno total, Sharpe Ratio, Max Drawdown, Win Rate.

### 5. Optimizacion de Portafolio
Frontera Eficiente de Markowitz con simulacion Montecarlo (5,000 portafolios):
- Portafolio de maximo Sharpe Ratio
- Portafolio de minimo riesgo
- Recomendacion segun perfil (conservador/moderado/agresivo)
- Matriz de correlacion entre activos

### 6. Base de Datos
Gestion de usuarios, portafolios y operaciones con SQLite:
- CRUD completo de usuarios con perfil de riesgo
- Portafolios con detalle de activos
- Historial de operaciones (compra/venta)
- Registro de predicciones de los modelos

## Estructura del Proyecto

```
proyecto_bolsa_streamlit/
├── app.py                          # Entrada principal + navegacion + CSS
├── config.py                       # Empresas, features, constantes
├── requirements.txt                # Dependencias
├── .streamlit/config.toml          # Configuracion del tema
│
├── views/                          # Paginas de la aplicacion
│   ├── dashboard.py                # Dashboard de mercado
│   ├── clasificacion.py            # Modelos de clasificacion
│   ├── regresion.py                # Modelos de regresion
│   ├── backtesting.py              # Backtesting de estrategias
│   ├── portafolio.py               # Optimizacion de portafolio
│   └── base_datos.py               # Gestion de BD
│
├── utils/                          # Modulos reutilizables
│   ├── data_loader.py              # Descarga de datos + indicadores tecnicos
│   ├── model_utils.py              # Carga de modelos entrenados
│   ├── backtesting_utils.py        # Motor de backtesting
│   └── portfolio_optimizer.py      # Optimizacion Markowitz
│
├── database/                       # Base de datos SQLite
│   ├── init_db.py                  # Creacion de tablas
│   ├── seed_data.py                # Datos de muestra
│   └── db_utils.py                 # Funciones CRUD
│
├── models/                         # Modelos entrenados (.pkl y .h5)
│   ├── svc/                        # SVC por empresa
│   ├── dl_clasificacion/           # SimpleRNN, LSTM, BiLSTM, GRU
│   └── regresion/                  # ARIMA, LSTM Regressor, ARIMA-LSTM
│
└── colab/                          # Notebooks de entrenamiento
    ├── 01_SVC_Clasificacion.ipynb
    ├── 02_DL_Clasificacion.ipynb
    └── 03_Regresion_Modelos.ipynb
```

## Instalacion y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/equipo01cienciadedatos2026/proyecto-bolsa-streamlit.git
cd proyecto-bolsa-streamlit
```

### 2. Crear y activar entorno virtual

```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Inicializar la base de datos

```bash
python -m database.init_db
python -m database.seed_data
```

### 5. Ejecutar la aplicacion

```bash
streamlit run app.py
```

## Tecnologias

| Categoria | Tecnologias |
|-----------|-------------|
| Frontend | Streamlit, Plotly, CSS |
| ML Clasico | scikit-learn (SVC, GridSearchCV) |
| Deep Learning | TensorFlow/Keras (RNN, LSTM, BiLSTM, GRU) |
| Series Temporales | statsmodels, pmdarima (ARIMA) |
| Datos | yfinance, pandas, numpy |
| Base de Datos | SQLite3 |
| Optimizacion | Simulacion Montecarlo (Markowitz) |

## Notas

- Los modelos de Deep Learning requieren **TensorFlow**, compatible con Python 3.10-3.12. En Python 3.14, estos modelos funcionan en modo demo (basado en señales tecnicas reales).
- El modelo **SVC** y **ARIMA** funcionan completamente en cualquier version de Python.
- Los modelos fueron entrenados en **Google Colab** usando los notebooks de la carpeta `colab/`.
