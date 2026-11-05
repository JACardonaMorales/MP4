# Predicción de Resultados de Fútbol mediante Redes Neuronales

## Descripción del proyecto

Este proyecto consiste en el desarrollo de una aplicación que utiliza una red neuronal artificial para predecir los resultados de partidos de fútbol. El modelo analiza diferentes características estadísticas de los equipos para clasificar los partidos en tres posibles resultados: victoria del equipo local, empate o victoria del equipo visitante.

## Objetivo

Implementar y evaluar un modelo de red neuronal tipo MLP (Multilayer Perceptron), explorando cómo diferentes características de los equipos y configuraciones de la red influyen en la capacidad de predicción.

## Dataset

El proyecto utiliza un conjunto de datos históricos de partidos de fútbol almacenado en el archivo `results.csv`. A partir de esta información se calculan diferentes atributos estadísticos, como la diferencia de puntuación Elo, el rendimiento reciente y los goles anotados y recibidos por cada equipo.

## Modelo de red neuronal

Se utiliza una red neuronal artificial de tipo MLP, compuesta por una capa de entrada, una o dos capas ocultas y una capa de salida con tres neuronas, correspondientes a los posibles resultados del partido.

El modelo permite configurar diferentes funciones de activación, tasas de aprendizaje, cantidades de neuronas, épocas de entrenamiento y tamaños de lote.

## Tecnologías utilizadas

El proyecto está desarrollado en Python y utiliza NumPy para las operaciones matemáticas, Pandas para el procesamiento de datos, Matplotlib para la generación de gráficas y Streamlit para la interfaz de usuario.

##  Guía de Instalación y Ejecución Paso a Paso

Sigue estas instrucciones en tu terminal para ejecutar el proyecto en tu máquina:

### 1. Abrir la terminal en la carpeta del proyecto
Abre una terminal (PowerShell, CMD, bash o zsh) y navega hasta la carpeta del proyecto:
```bash
cd ruta/hacia/MP4
```

### 2. Crear y activar un entorno virtual

* **En Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nota: Si PowerShell muestra un error de políticas de script, ejecuta `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve a activar).*

* **En Windows (Símbolo del sistema / CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

* **En macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar dependencias
Con el entorno virtual activado, instala los paquetes requeridos:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
Ejecuta el servidor de Streamlit con el siguiente comando:
```bash
streamlit run app.py
```

Al iniciarse, se abrirá automáticamente una pestaña en tu navegador en la dirección local:
 **`http://localhost:8501`**

---

## Funcionalidades

La aplicación permite seleccionar los atributos de entrada, configurar los hiperparámetros de la red neuronal, entrenar el modelo y visualizar sus resultados.

También permite realizar comparaciones entre diferentes atributos y combinaciones de características para analizar su influencia en el rendimiento del modelo.
