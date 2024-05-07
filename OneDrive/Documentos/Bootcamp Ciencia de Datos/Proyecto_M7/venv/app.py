from flask import Flask, request, jsonify
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import traceback

app = Flask(__name__)

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Función para preprocesar el texto
def preprocess_text(text):
    # Convertir texto a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Eliminar palabras vacías (stopwords)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematizar las palabras
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Unir tokens en una sola cadena
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

@app.route('/analiza_sentimiento', methods=['POST'])
def predict_sentiment():
    try:
        # Obtener los datos de la solicitud
        data = request.get_json(force=True)
        
        # Preprocesar el texto
        preprocessed_text = preprocess_text(data['review'])
        
        # Cargar el modelo entrenado
        model = joblib.load('sentiment_analysis_model.pkl')
        
        # Cargar el vectorizador TF-IDF ajustado
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        # Transformar el texto preprocesado en una matriz 2D
        preprocessed_text_matrix = tfidf_vectorizer.transform([preprocessed_text])
        
        # Realizar la predicción
        prediction = model.predict(preprocessed_text_matrix)
        
        # Calcular la probabilidad de cada clase para el texto de prueba
        probabilities = model.predict_proba(preprocessed_text_matrix)[0]

        # Obtener la probabilidad de la clase específica según la etiqueta predicha
        if prediction[0] == 'Negative':
            sentiment = 'Negativo'
            probability = probabilities[model.classes_ == "Negative"][0]
            probability_per = np.round(probability * 100)
        elif prediction[0] == 'Positive':
            sentiment = 'Positivo'
            probability = probabilities[model.classes_ == "Positive"][0]
            probability_per = np.round(probability * 100)
        elif prediction[0] == 'Neutral':
            sentiment = 'Neutral'
            probability = probabilities[model.classes_ == "Neutral"][0]
            probability_per = np.round(probability * 100)
        
        output = {'Sentimiento predicho': sentiment, 'Probabilidad': probability_per}
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})

if __name__ == '__main__':
    app.run(port=8000, debug=True)

