# 📈 Prototipo BI + IA Aplicada a Bolsa (MVP)

Este proyecto es un Producto Mínimo Viable (MVP) desarrollado con **Streamlit** para visualizar y analizar datos financieros de activos mineros y globales utilizando la API de **Yahoo Finance**.

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

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/equipo01cienciadedatos2026/proyecto-bolsa-streamlit.git](https://github.com/equipo01cienciadedatos2026/proyecto-bolsa-streamlit.git)
   cd proyecto-bolsa-streamlit

2. **Crear y activar entorno virtual:**
python -m venv .venv
# En Windows:
.\.venv\Scripts\activate   

3. **Instalar dependencias:**
pip install -r requirements.txt

4. **Ejecutar Aplicacion**
streamlit run app.py