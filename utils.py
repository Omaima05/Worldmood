import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from transformers import pipeline
import os

# === Connexion MongoDB ===
def connect_to_mongo(uri: str = None, db_name: str = "worldmood", articles: str = "articles"):
    """
    Connexion à la base MongoDB.
    """
    uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(uri)
    db = client["articles_db"]
    return db["articles"]

# === Récupérer les articles nettoyés depuis MongoDB ===
def get_cleaned_articles():
    """
    Récupère les articles avec contenu depuis la collection MongoDB.
    """
    collection = connect_to_mongo()
    articles = list(collection.find({"content": {"$exists": True}}))
    return articles

# === Chargement du modèle d’analyse émotionnelle ===
def load_emotion_model():
    """
    Charge un pipeline HuggingFace pour la classification émotionnelle.
    """
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# === Analyse émotionnelle sur un DataFrame ===
def analyze_emotions(df: pd.DataFrame, emotion_analyzer) -> pd.DataFrame:
    """
    Applique l'analyse émotionnelle sur les textes du DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data is not a valid DataFrame.")

    df = df.copy()

    # Normalisation des timestamps
    def parse_timestamp(ts):
        if isinstance(ts, datetime):
            return ts
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime.utcnow()

    if "timestamp" not in df.columns or df["timestamp"].isnull().all():
        df["timestamp"] = datetime.utcnow()
    else:
        df["timestamp"] = df["timestamp"].apply(parse_timestamp)

    # Nettoyage des colonnes
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["year"] = df["timestamp"].dt.year
    df["date"] = df["timestamp"].dt.date.astype(str)

    if "theme" not in df.columns:
        df["theme"] = "Inconnu"

    if "pays" in df.columns:
        df["country"] = df["pays"]
    elif "country" not in df.columns:
        df["country"] = "Inconnu"

    # Analyse émotionnelle (sur les 512 premiers caractères)
    if "emotion" not in df.columns:
        if "content" in df.columns:
            df["emotion"] = df["content"].apply(
                lambda text: emotion_analyzer(text[:512])[0]["label"]
                if isinstance(text, str) and text.strip()
                else "unknown"
            )
        else:
            df["emotion"] = "unknown"

    df["nb_articles"] = 1

    return df
