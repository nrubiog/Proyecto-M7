# Análisis de Sentimientos en Reseñas de Aplicaciones de la Google Play Store

Este proyecto utiliza procesamiento de lenguaje natural (PLN) para realizar un análisis de sentimientos a las reseñas de las aplicaciones de Google Play Store. Bajo esta idea, se ha desarrollado un modelo de clasificación que puede predecir si una reseña es positiva, neutral o negativa.

## Descripción del Proyecto

El proyecto consta de los siguientes componentes:

1. **Dataset**: Se utiliza un conjunto de datos que contiene reseñas de aplicaciones de la Google Play Store, junto con sus calificaciones y otros detalles, extraido de Kaggle.

2. **Preprocesamiento de Datos**: Se realizan tareas de limpieza y preprocesamiento en el texto de las reseñas, que incluyen la eliminación de caracteres especiales, tokenización, eliminación de stopwords y lematización.

3. **Entrenamiento del Modelo**: Se entrena un modelo de clasificación utilizando algoritmos de aprendizaje automático como Naive Bayes.

4. **Despliegue de la API**: Se crea una API web utilizando Flask que permite enviar reseñas y recibir la predicción de sentimiento correspondiente.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

- **data/**: Contiene los conjuntos de datos utilizados para el entrenamiento y la evaluación del modelo.
- **models/**: Almacena los modelos entrenados y los objetos necesarios para la vectorización de texto.
- **app.py**: Archivo principal de la API Flask.
- **sentiment_analysis_model.pkl**: Modelo entrenado para la clasificación de sentimientos.
- **tfidf_vectorizer.pkl**: Objeto vectorizador TF-IDF para la transformación de texto.
- **README.md**: Este archivo que proporciona información sobre el proyecto.

## Uso de la API

Para utilizar la API, envía una solicitud POST a la ruta `/analiza_sentimiento` con el siguiente formato JSON:

```json
{
  "review": "Tu reseña o comentario."
}
```
La API devolverá una respuesta con el sentimiento predicho y su probabilidad asociada, por ejemplo:

```json
{
  "Sentimiento": "Positiva",
  "Probabilidad": "80%"
}
```

