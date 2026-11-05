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

## Ejecución del proyecto

Para ejecutar la aplicación es necesario tener Python instalado y las dependencias del proyecto disponibles.

Primero, se deben instalar las dependencias necesarias y posteriormente ejecutar la aplicación mediante Streamlit.

## Funcionalidades

La aplicación permite seleccionar los atributos de entrada, configurar los hiperparámetros de la red neuronal, entrenar el modelo y visualizar sus resultados.

También permite realizar comparaciones entre diferentes atributos y combinaciones de características para analizar su influencia en el rendimiento del modelo.