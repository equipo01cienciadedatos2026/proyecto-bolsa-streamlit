# 📈 Prototipo BI + IA Aplicada a Bolsa (MVP)



Este proyecto es un Producto desarrollado con **Streamlit** para visualizar y analizar datos financieros de activos mineros y globales utilizando la API de **Yahoo Finance**.



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

git clone [https://github.com/equipo01cienciadedatos2026/proyecto-bolsa-streamlit.git](https://github.com/equipo01cienciadedatos2026/proyecto-bolsa-streamlit.git)

cd proyecto-bolsa-streamlitahora



### 2. Crear y activar entorno virtual



```powershell

# Crear el entorno virtual

python -m venv .venv



# Activar el entorno en Windows (PowerShell):

.\.venv\Scripts\Activate.ps1



### 3. Instalar dependencias



Una vez activado el entorno, instala todas las librerías necesarias (Streamlit, Pandas, YFinance, etc.) usando el archivo de requerimientos:



```bash

pip install -r requirements.txt





### 4. Ejecutar la Aplicación



Finalmente, inicia el servidor local de Streamlit para ver el prototipo en tu navegador:



```bash

streamlit run app.py