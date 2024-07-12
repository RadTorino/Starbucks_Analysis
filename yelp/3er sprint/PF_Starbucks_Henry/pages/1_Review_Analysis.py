import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib
from joblib import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.exceptions import NotFittedError

# Path de los modelos preentrenados
MANIPULATED_PATH = 'models/manipulated_reviews.pkl'
SENTIMENT_PATH = 'models/sentiment_analysis.pkl'
VECTORIZER_PATH = 'models/rating_vectorizer.pkl'
CLUSTER_PATH = 'models/cluster_analysis.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'

# Cargar los modelos guardados
loaded_model_MANIPULATED = joblib.load(MANIPULATED_PATH)
loaded_model_SENTIMENT = joblib.load(SENTIMENT_PATH)
pipeline_VECTORIZER = joblib.load(VECTORIZER_PATH)
loaded_TFIDF_MANIPULATED = joblib.load(TFIDF_PATH)


# Título
html_temp = """
<h2 style="color:#006847;text-align:center;">Machine Learning Project for Business Performance Optimization</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

# CSS para estilizar el botón
button_style = """
<style>
.custom-button {
    background-color: #FFFFFF;
    color: #000000;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    font-weight: bold;
}
</style>
"""
# Renderiza el CSS
st.markdown(button_style, unsafe_allow_html=True)

# TOKENIZADO ______________________________________________________________________________________

# Descargar recursos de nltk
nltk.download('punkt')
nltk.download('stopwords')

# Definir la función para normalizar, tokenizar y capitalizar el texto
def process_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Remover palabras vacías
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    # Capitalizar cada palabra
    tokens = [word.capitalize() for word in tokens]
    # Unir tokens en una cadena de texto
    processed_text = ' '.join(tokens)
    return processed_text    

# MANIPULATED_PATH _________________________________________________________________________________

def predict_review(text, stars):
    # Transformar el texto usando el vectorizador TF-IDF
    text_tfidf = loaded_TFIDF_MANIPULATED.transform([text])

    # Combinar las características de texto con la columna stars
    combined_features = np.hstack((text_tfidf.toarray(), [[stars]]))

    # Hacer una predicción
    prediction = loaded_model_MANIPULATED.predict(combined_features)

    # Devolver el resultado
    return 'False' if prediction[0] == 1 else 'Genuine'

# SENTIMENT_PATH ___________________________________________________________________________________

# Definir un diccionario que mapea los valores de sentimiento a emojis
sentiment_to_emoji = {
    0: "😞",  # negativo
    1: "😐",  # neutral
    2: "😊"   # positivo
}

# Función para predecir y devolver el emoji correspondiente
def predict_sentiment_SENTIMENT(text):
    prediction = loaded_model_SENTIMENT.predict([text])[0]
    return sentiment_to_emoji[prediction]

# VECTORIZER_PATH __________________________________________________________________________________

# Definir la función para predecir el rating del negocio
def predict_rating_VECTORIZER(text, stars):
    # Hacer una predicción del rating del negocio
    predicted_rating = pipeline_VECTORIZER.predict([text])[0]

    # Mostrar el resultado
    st.write(f'The predicted rating of the business based on the review is: {predicted_rating:.2f}')

    # Hacer recomendaciones según el rating predicho
    if predicted_rating >= 4.5:
        st.write("Business recommendation: Business Rating: Excellent service! Keep focused on maintaining high quality.")
    elif 3.5 <= predicted_rating < 4.5:
        st.write("Business recommendation: Here are some areas for improvement based on the reviews received:")
        # Puedes imprimir recomendaciones específicas basadas en el análisis de las revisiones aquí.
    else:
        st.write("Business recommendation:  We have identified critical areas that require urgent attention:")
        # Puedes imprimir recomendaciones específicas para abordar los problemas identificados aquí.

# CLUSTER_PATH _____________________________________________________________________________________

# Crear un DataFrame estático con datos de ejemplo
data = {
    'text': [
            "Ordered Caramel frappe at Drive thru, BIG MISTAKE! Took 30 min and there were only 2 cars in front of me.",
            "Drum-roll please! Review #100 coming right up! \n\nI chose to review Starbucks as my 100th review because it is a guilty pleasure. Something that I only allow myself once a week. But something that I crave everyday! \n\nThis particular Starbucks is great. The baristas are always friendly and welcoming. The drinks are made correctly and quickly. I usually go with a Salted Caramel Hot Chocolate or Mocha. I also occasionally order the White Mocha. If you can't tell, I enjoy coffee that doesn't taste like coffee! In the summer their iced teas are the perfect thirst quencher! Their breakfast pastry items are always delicious. Try the Cranberry-Orange Scone and you will not be disappointed! Their breakfast sandwiches are unique and pretty tasty as well.\n\nI always choose to go in and place my order over sitting in the drive thru. The line for their drive thru is always ridiculously long. It's usually quicker to go in.",
            "We stopped here for my Chai and Hubby's coffee and caffeine mania. The line went quite fast, they're quick with the service especially not a lot of people at the time. My Chai came out good, awesome! One thing I forgot to do is to purchase more coffee for the house. They have lots for sale. But not too worry because we still have plenty of Starbucks coffee at home. What would life be without Starbucks?.",
            "There's been three times that I've ordered a green tea lemonade and got a peach tea lemonade and had to turn all the way back and go inside and have theme remake it",
            "I went in when they had 4 people working, waited for 15 minutes for my tea latte which came out as basically water. Returned it twice but gave up when the barista offered some other iced tea latte they already had instead. It was still the same - water. The baristas should REALLY REALLY learn how to make their stuff... seriously Starbucks??!!! So disappointed!!!!! :'(.",
            " Most of the time I go through the drive thru here.  While sometimes the line is a bit long I have to say these guys move in record time.  The line moves fast and my coffee is waiting for me when I get to the window.  How much more can you ask for? \n\nInside there is a decent amount of seating and there is quite a bit of outdoor seating too.  Overall this is a great Starbuck",
            """i dont know what has happened to the in store service in this place!  We have been down here for a few weeks now and have been here 6 times.Only once was the order completed without a screw up!\n\nToday 2 tall decaf coffees took close to 20 minutes.By then our bagels were cold.First i was told the decaf was "brewing' and then i was told there was something 'wrong" with the brewer? After 15 minutes i was offered a decaf americano.\nBefore we left i politely told a fellow behind the counter the service was abominable and he said 'well we can only store 2 vente size decafs at a time? HUH? I asked if the management knew about this and he just shrugged.\nClearly this staff has never been trained to make it right for the customer.No apology.No coupon.Just a convoluted story.\n \nAmazing how you can have numerous people running around a store but you cant get the simplest order straight My next visit to a Starbucks will certainly not be this one""",
            "Nothing makes my busy day easy like my iced coffee at Starbucks and this location botched it up for me: see my photo yup those are coffee grinds at the bottom a whole bunch of coffee grinds:( yuck yuck!"
    ],
    'stars': [2.0, 4.0, 4.0, 2.0, 1.0, 4.0, 1.0, 1.0]
}

df_train = pd.DataFrame(data)

@st.cache_resource
def load_model_and_preprocessor():
    kmeans = load(CLUSTER_PATH)  # Cargar el modelo KMeans
    preprocessor = load(PREPROCESSOR_PATH)  # Cargar el vectorizador
    return kmeans, preprocessor

# Función para predecir clusters
def predict_cluster(new_text, new_stars, kmeans, preprocessor):
    new_data = {'stars': [float(new_stars)], 'text': [new_text]}
    df_new_data = pd.DataFrame(new_data)

    # Preprocesar los nuevos datos
    X_new_scaled = preprocessor.transform(df_new_data['text'])

    # Predecir clusters
    predicted_clusters_new = kmeans.predict(X_new_scaled)

    # Añadir los clusters predichos al DataFrame
    df_new_data['predicted_cluster'] = predicted_clusters_new

    return df_new_data


# LECTURA DE DATOS __________________________________________________________________________________

# Interfaz de Streamlit
def main():
    new_text = st.text_input("Text: enter the values for review Text")
    new_stars = st.text_input("Stars: decimal number from 1.0 to 5.0")
    
    # Datos de entrenamiento para el vectorizador TF-IDF estaticos
    X_train = [
                "Ordered Caramel frappe at Drive thru, BIG MISTAKE! Took 30 min and there were only 2 cars in front of me.",
                "Drum-roll please! Review #100 coming right up! \n\nI chose to review Starbucks as my 100th review because it is a guilty pleasure. Something that I only allow myself once a week. But something that I crave everyday! \n\nThis particular Starbucks is great. The baristas are always friendly and welcoming. The drinks are made correctly and quickly. I usually go with a Salted Caramel Hot Chocolate or Mocha. I also occasionally order the White Mocha. If you can't tell, I enjoy coffee that doesn't taste like coffee! In the summer their iced teas are the perfect thirst quencher! Their breakfast pastry items are always delicious. Try the Cranberry-Orange Scone and you will not be disappointed! Their breakfast sandwiches are unique and pretty tasty as well.\n\nI always choose to go in and place my order over sitting in the drive thru. The line for their drive thru is always ridiculously long. It's usually quicker to go in.",
                "We stopped here for my Chai and Hubby's coffee and caffeine mania. The line went quite fast, they're quick with the service especially not a lot of people at the time. My Chai came out good, awesome! One thing I forgot to do is to purchase more coffee for the house. They have lots for sale. But not too worry because we still have plenty of Starbucks coffee at home. What would life be without Starbucks?.",
                "There's been three times that I've ordered a green tea lemonade and got a peach tea lemonade and had to turn all the way back and go inside and have theme remake it",
                "I went in when they had 4 people working, waited for 15 minutes for my tea latte which came out as basically water. Returned it twice but gave up when the barista offered some other iced tea latte they already had instead. It was still the same - water. The baristas should REALLY REALLY learn how to make their stuff... seriously Starbucks??!!! So disappointed!!!!! :'(.",
                " Most of the time I go through the drive thru here.  While sometimes the line is a bit long I have to say these guys move in record time.  The line moves fast and my coffee is waiting for me when I get to the window.  How much more can you ask for? \n\nInside there is a decent amount of seating and there is quite a bit of outdoor seating too.  Overall this is a great Starbuck",
                """i dont know what has happened to the in store service in this place!  We have been down here for a few weeks now and have been here 6 times.Only once was the order completed without a screw up!\n\nToday 2 tall decaf coffees took close to 20 minutes.By then our bagels were cold.First i was told the decaf was "brewing' and then i was told there was something 'wrong" with the brewer? After 15 minutes i was offered a decaf americano.\nBefore we left i politely told a fellow behind the counter the service was abominable and he said 'well we can only store 2 vente size decafs at a time? HUH? I asked if the management knew about this and he just shrugged.\nClearly this staff has never been trained to make it right for the customer.No apology.No coupon.Just a convoluted story.\n \nAmazing how you can have numerous people running around a store but you cant get the simplest order straight My next visit to a Starbucks will certainly not be this one""",
                "Nothing makes my busy day easy like my iced coffee at Starbucks and this location botched it up for me: see my photo yup those are coffee grinds at the bottom a whole bunch of coffee grinds:( yuck yuck!"
            ]
    y_train = [0, 1, 2, 3, 4, 5, 6, 7]  # Aquí deberías tener las etiquetas correspondientes
    
    if st.button("Predict"):
        if new_text.strip() == '' or not new_stars.replace('.', '', 1).isdigit() or not (1 <= float(new_stars) <= 5):
            st.warning("Please enter a review text.")
            
        else:
            # TOKENIZED ________________________
            # Ejecutar función de TOKENIZED si es necesario
            processed_text = process_text(new_text)
            st.write(f"Processed Text: {processed_text}")
            
            # MANIPULATED ______________________
            # Ejecutar función de MANIPULATED si es necesario
            result = predict_review(new_text, new_stars)
            st.write(f'The new review is {result}')
            
            # SENTIMENT ________________________
            # ejecutar la predicción    
            predicted_emoji = predict_sentiment_SENTIMENT(new_text)
            st.success(f"Predicted sentiment emoji: {predicted_emoji}")
            
            # VECTORIZER _______________________
            # ejecutar la predicción
            predict_rating_VECTORIZER(new_text, new_stars)

            # CLUSTER __________________________
            kmeans, preprocessor = load_model_and_preprocessor()
            if new_text and new_stars:
                # Realizar la predicción del cluster
                df_result = predict_cluster(new_text, new_stars, kmeans, preprocessor)

                # Mostrar los resultados
                st.write("Prediction Results Based on Cluster Analysis:")
                st.write(df_result)

                # Graficar los datos nuevos
                plt.figure(figsize=(14, 1.5))
                plt.scatter(df_result['stars'], [0] * len(df_result), c=df_result['predicted_cluster'], cmap='viridis')
                plt.xlabel('Stars', fontsize=10)  # Cambia el tamaño de la fuente aquí
                plt.title('Review Clusters for New Data', fontsize=12)  # Cambia el tamaño de la fuente aquí

                # Crear la colorbar
                cbar = plt.colorbar(label='Cluster')
                cbar.ax.tick_params(labelsize=8)  # Cambia el tamaño de la fuente para los ticks de la colorbar

                # Cambiar el tamaño de la fuente de la etiqueta de la colorbar
                cbar.set_label('Cluster', fontsize=8)  # Cambia el tamaño de la fuente aquí

                plt.xticks(fontsize=8)  # Cambia el tamaño de la fuente para las etiquetas del eje x
                plt.yticks(fontsize=8)  # Cambia el tamaño de la fuente para las etiquetas del eje y
                st.pyplot(plt)


if __name__ == "__main__":
    main()
