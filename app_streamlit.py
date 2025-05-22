 import streamlit as st
import pandas as pd
import plotly.express as px
from code_final import get_cleaned_articles, load_emotion_model, analyze_emotions, bubble_map
from pymongo import MongoClient

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["articles_db"]
collection = db["articles"]

def main():
    st.title("WorldMood – Analyse et visualisation des émotions")

    # Charger les articles depuis MongoDB
    articles = get_cleaned_articles()
    if not articles:
        st.warning("Aucun article disponible dans la base.")
        return

    df = pd.DataFrame(articles)

    # Nettoyage et préparation des colonnes nécessaires
    if "timestamp" in df.columns and "year" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["year"] = df["timestamp"].dt.year.fillna(0).astype(int)

    # Nettoyage colonne 'theme'
    if "theme" not in df.columns:
        df["theme"] = "Inconnu"
    else:
        # Remplacer valeurs manquantes, forcer str
        df["theme"] = df["theme"].fillna("Inconnu").astype(str)

    # Affichage diagnostic
    st.write("Types dans df['theme'] :", df["theme"].apply(type).value_counts())
    st.write("Valeurs uniques dans df['theme'] :", df["theme"].unique())

    # Nettoyage colonne 'emotion'
    if "emotion" not in df.columns or df["emotion"].isnull().all():
        emotion_model = load_emotion_model()
        df = analyze_emotions(df, emotion_model)
    else:
        # S'assurer que ce sont bien des strings et pas de valeurs manquantes
        df["emotion"] = df["emotion"].fillna("neutral").astype(str)

    # Sélections dans la sidebar
    selected_theme = st.sidebar.selectbox("Choisir un thème", options=df["theme"].unique())
    selected_emotion = st.sidebar.selectbox("Choisir une émotion", options=df["emotion"].unique())

    # Slider année
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    st.write(f"Années disponibles : {min_year} → {max_year}")

    if min_year == max_year:
        selected_year = min_year
        st.info(f"Les articles sont uniquement de l'année {selected_year}.")
    else:
        selected_year = st.sidebar.slider("Choisir l'année", min_year, max_year, min_year)

    # Affichage de la carte à bulles
    try:
        fig = bubble_map(df, selected_theme, selected_emotion, selected_year)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de la carte : {e}")

if __name__ == "__main__":
    main()

# Partie comparaison des émotions par pays (après main)

# Liste complète des émotions
emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

# Récupération des pays
pays_list = collection.distinct("pays")

st.title("Comparaison des émotions par pays")

selected_countries = st.multiselect(
    "Choisir 3 pays :", pays_list, default=pays_list[:3]
)

if len(selected_countries) != 3:
    st.warning("Veuillez sélectionner exactement 3 pays.")
else:
    data = []
    for pays in selected_countries:
        articles = collection.find({"pays": pays})

        counts = {e: 0 for e in emotions}
        for article in articles:
            emotion = article.get("emotion")
            if emotion in emotions:
                counts[emotion] += 1

        for emotion in emotions:
            data.append({
                "Pays": pays,
                "Emotion": emotion,
                "Nombre d'articles": counts[emotion]
            })

    df_pays = pd.DataFrame(data)

    fig = px.bar(
        df_pays,
        x="Pays",
        y="Nombre d'articles",
        color="Emotion",
        category_orders={"Emotion": emotions},
        barmode="stack",
        title="Émotions par pays",
        color_discrete_map={
            "joy": "#E1952B",
            "sadness": "#4282BD",
            "anger": "#AD211F",
            "fear": "#AB6FCB",
            "surprise": "#E15F88",
            "disgust": "#0E7B0E",
            "neutral": "#AE9F9F"
        }
    )

    st.plotly_chart(fig, use_container_width=True)


