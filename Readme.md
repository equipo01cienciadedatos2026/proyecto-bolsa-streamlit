# 📈 Prototipo BI + IA Aplicada a Bolsa (MVP)



Este proyecto es un Producto desarrollado con **Streamlit** para visualizar y analizar datos financieros de activos mineros y globales utilizando la API de **Yahoo Finance**.

## Grupo 1
### Integrantes

* Asencios Rojas, Herberth Alvaro.
* Benites Meza, Marco Fabricio.
* Del Solar Rojas, Jorge Sebastian.
* Guerrero Jaramillo, Andres Abraham.
* Peralta Farfán, Raymond Alain.
* Matos Ramos, Franco Antonio.
* Herrera Fernandez Yumerth Mijail.
* Vidalon Flores, Daniel Omar.
* Rojas Huamaní, Percy Ares.



## 🚀 Características

* **Conexión en Tiempo Real:** Extracción de datos financieros actualizados mediante `yfinance`.

* **Visualización Interactiva:** Gráficos dinámicos de precios de cierre construidos con `Plotly Express`.

* **Interfaz BI:** Diseño limpio con selección de activos y rangos de fechas personalizados.

* **Optimización:** Uso de `st.cache_data` para mejorar el rendimiento en la carga de datos.



## 🛠️ Tecnologías Utilizadas

* **Python 3.x**

* **Streamlit** (Interfaz de usuario)

* **Pandas** (Manipulación de datos)

* **Plotly** (Gráficos interactivos)

* **YFinance** (Fuente de datos financieros)


## 📦 Instalación y Uso

### 1. Clonar el repositorio

```bash
 git clone https://github.com/equipo01cienciadedatos2026/proyecto-bolsa-streamlit.git
```
### luego
```bash
cd proyecto-bolsa-streamlit
```

### 2. Crear y activar entorno virtual
```bash
python -m venv .venv
```
### Activar el entorno en Windows (PowerShell):
```bash
.\.venv\Scripts\Activate.ps1
```


### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```


### 4. Crear y activar entorno virtual
```bash
streamlit run app.py
```


