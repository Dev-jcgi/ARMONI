# MindWave EEG Data Visualization Tools

## Descripción
Sistema de visualización y análisis de datos EEG para el dispositivo MindWave Mobile 2.

## Estructura del Proyecto

mindwave-eeg/
├── app.py                 # Aplicación Streamlit
├── rec.py                # Script de grabación
├── RealTime.ipynb        # Notebook Jupyter
├── requirements.txt      # Dependencias
├── data/                 # Datos
│   └── recordings/      # Grabaciones EEG
└── docs/                # Documentación

## Requisitos
- Python 3.7+
- MindWave Mobile 2
- Windows/macOS/Linux
- Puerto COM/Serial

## Instalación

### Entorno Virtual
```bash
# Crear entorno
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate

## Dependencias
### Instalar requerimientos
```bash
pip install -r requirements.txt

### Fix pyserial
```bash
pip uninstall serial
pip install pyserial

## Uso
### Streamlit (app.py)
```bash
streamlit run app.py


## Características:

--Visualización en tiempo real

--Análisis de atención/meditación

--Filtrado de señales

--Exportación CSV

--Grabación (rec.py)

```bash
python rec.py --port COM3 --duration 60 --output data/recordings

Parámetros:

--port: Puerto COM
--duration: Duración (segundos)
--output: Directorio salida

# Jupyter (RealTime.ipynb)

## jupyter notebook
Abrir RealTime.ipynb y seguir instrucciones.

Configuración
Filtros

# app.py
self.fs = 512  # Frecuencia muestreo
self.nyquist = self.fs / 2
self.b, self.a = signal.butter(2, [0.5/self.nyquist, 30/self.nyquist], btype='band')

Visualización
self.display_points = 200  # Puntos en gráfica
self.update_interval = 0.1  # Intervalo actualización


Solución Problemas

Conexión
# Listar puertos
python -m serial.tools.list_ports

# Permisos Linux/Mac
sudo chmod 666 /dev/ttyUSB0

Visualización

# Limpiar cache
streamlit cache clear
Datos
Formato CSV
CSV

timestamp,attention,meditation,raw,delta,theta,alpha_low,alpha_high,beta_low,beta_high,gamma_low,gamma_mid

Ubicación

Streamlit: ~/Documents/eegdata/
Grabación: data/recordings/
Jupyter: Directorio actual

Visualizaciones
Streamlit
Gráficas tiempo real
Métricas atención/meditación
Análisis ondas cerebrales
Calidad señal
Jupyter

Análisis detallado
Gráficas interactivas
Procesamiento personalizado
Exportación resultados
Grabación
Datos raw
Métricas básicas
Timestamps
CSV compatible
Desarrollo
Contribuir
Fork repositorio
Crear rama
Commit cambios
Push rama
Pull Request
Local
BASH

# Entorno desarrollo
pip install -r requirements-dev.txt

# Tests
pytest

# Formato
black .
Enlaces
MindWave Docs
Streamlit
Jupyter
Licencia
MIT

Soporte
GitHub Issues
Pull Requests
Contacto equipo
Última actualización: [fecha]
