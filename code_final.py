import requests
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline)
from datetime import datetime
import geocoder
from deep_translator import GoogleTranslator
import warnings
import re
import nltk
import plotly.express as px
import plotly.graph_objects as go
import json
from bson.json_util import dumps

warnings.filterwarnings("ignore")
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('french'))

# === CONFIGURATION ===
NEWSAPI_URL = "https://newsapi.org/api/v4/Everything"
NEWSAPI_KEY = "5cc959f32-de5449ea982ea306e99cd0d"
USE_CUSTOM_MODEL = False  # Toggle modèle personnalisé

# === CHARGEMENT DES PIPELINES ===
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                            return_all_scores=True)

try:
    custom_model = pipeline("text-classification", model="./emotion_model", tokenizer="./emotion_model")
except:
    custom_model = None

# === OUTILS NLP pour le nettoyage ===
tokenizer_cleaner = AutoTokenizer.from_pretrained("camembert-base")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)


# === MONGODB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["articles_db"]
collection = db["articles"]


# 🔁 Étape unique : convertir "country" -> "pays" si besoin
def migrate_country_to_pays():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["articles_db"]
    collection = db["articles"]
    result = collection.update_many(
        {"country": {"$exists": True}},
        [{"$set": {"pays": "$country"}}, {"$unset": "country"}]
    )
    print(f"🛠️ Migration terminée : {result.modified_count} article(s) mis à jour (country ➜ pays)")


# === TRADUCTION ===
def translate_to_english(text, source_lang):
    if not text or source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except:
        return text


# Récupérer les articles depuis NewsAPI en paginant
def get_newsapi_articles(country, language, query="", max_pages=5):
    params = {
        "q": query,
        "language": language,
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY
    }
    all_articles = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        response = requests.get(NEWSAPI_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            all_articles.extend(articles)
            if not articles:
                break
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            break
    return all_articles


def save_articles_to_mongo(articles):
    count = 0
    if not articles:
        print("⚠️ Aucun article à sauvegarder.")
        return
    for a in articles:
        if not collection.find_one({"url": a["url"]}):
            collection.insert_one(a)
            count += 1
    print(f"{count} article(s) ajouté(s) dans MongoDB.")


emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

for doc in collection.find({"emotion": {"$exists": False}}):
    # Texte prioritaire : content > title_en > title_original
    text = doc.get("content") or doc.get("title_en") or doc.get("title_original")

    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        print(f"⚠️ Aucun texte exploitable pour {doc.get('_id')}")
        continue

    try:
        result = emotion_model(text[:512])
        # Parfois le résultat est [[...]], parfois juste [...]
        if isinstance(result[0], list):
            emotion = result[0][0]
        else:
            emotion = result[0]

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "emotion": emotion["label"],
                "score_emotion": float(emotion["score"])
            }}
        )
        print(f"✅ Émotion ajoutée pour : {text[:60]}...")

    except Exception as e:
        print(f"❌ Erreur pour {doc.get('_id')} → {e}")


# === EMOTIONS ===
def detect_emotion_custom(text):
    if custom_model is None:
        raise ValueError("Le modèle personnalisé n'est pas chargé.")
    return max(custom_model(text[:512]), key=lambda x: x['score'])["label"]


def detect_emotion(text):
    result = emotion_pipeline(text[:512])[0]
    return max(result, key=lambda x: x["score"])["label"]


def detect(text):
    if USE_CUSTOM_MODEL:
        if custom_model is not None:
            label = detect_emotion_custom(text)
            score = None
        else:
            print("⚠️ Custom model non chargé, utilisation du modèle standard")
            result = emotion_pipeline(text[:512])[0]
            sorted_res = sorted(result, key=lambda x: x["score"], reverse=True)
            label = sorted_res[0]["label"]
            score = round(sorted_res[0]["score"], 3)
    else:
        result = emotion_pipeline(text[:512])[0]
        sorted_res = sorted(result, key=lambda x: x["score"], reverse=True)
        label = sorted_res[0]["label"]
        score = round(sorted_res[0]["score"], 3)
    return label, score


def annotate_articles_with_emotions():
    for article in collection.find({"emotion": {"$exists": False}}):
        original_text = article.get("content") or article.get("description") or ""
        if not original_text:
            continue
        lang = article.get("lang", "en")
        translated_text = translate_to_english(original_text, source_lang=lang)
        if not translated_text.strip():
            continue
        emotion_label, emotion_score = detect(translated_text)
        couleur_map = {
            "joy": "jaune", "sadness": "bleu", "anger": "rouge",
            "fear": "violet", "love": "rose", "surprise": "orange"
        }
        couleur = couleur_map.get(emotion_label, "gris")
        pays = geocoder.ip('me').country or "Inconnu"
        collection.update_one(
            {"_id": article["_id"]},
            {"$set": {
                "emotion": emotion_label,
                "score_emotion": emotion_score,
                "translated_text": translated_text,
                "couleur": couleur,
                "pays": pays,
                "timestamp": datetime.utcnow().isoformat()
            }}
        )
        print(f"✅ {emotion_label} ({lang} ➜ en | {pays}) - {article['title']}")


# === MODEL TRAINING ===
def train_custom_model(texts, labels, model_name="distilbert-base-uncased"):
    df = pd.DataFrame({"text": texts, "label": labels})
    df["cleaned_text"] = df["text"].apply(clean_text)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_dataset = train_dataset.map(lambda b: tokenizer(b["cleaned_text"], padding=True, truncation=True),
                                      batched=True)
    val_dataset = val_dataset.map(lambda b: tokenizer(b["cleaned_text"], padding=True, truncation=True), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))
    args = TrainingArguments(output_dir="./emotion_model", num_train_epochs=3, per_device_train_batch_size=4)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
                      tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained("./emotion_model")
    tokenizer.save_pretrained("./emotion_model")


# === VISUALISATION ===
def plot_emotions():
    emotions = [a["emotion"] for a in collection.find({"emotion": {"$exists": True}})]
    if not emotions:
        raise ValueError("Aucune donnée à afficher. Vérifie que la base contient des émotions annotées.")
    counts = Counter(emotions)
    plt.bar(counts.keys(), counts.values(), color="orange")
    plt.title("Distribution des émotions")
    plt.xlabel("Émotion")
    plt.ylabel("Articles")
    plt.show()


def show_articles_by_emotion(emotion):
    print(f"\nArticles avec l'émotion : {emotion}")
    for a in collection.find({"emotion": emotion}):
        print(f"• {a['title']} ({a['source']})")


# === CONFIGURATION MULTI-THEMES / MULTI-LANGUES ===
THEMES = ["politique", "Ukraine", "Guerre", "culture", "économie", "Israël"]
LANGUAGES = ["fr", "en", "de", "he", "ze", "en", "es", "it", "pt", "ru", "zh", "ar", "fa"]
TOPICS_BY_COUNTRY = {
    "France": ("fr", "politique OR économie OR Ukraine OR Israël"),
    "USA": ("en", "election OR Trump OR economy OR Gaza"),
    "Germany": ("de", "wirtschaft OR politik OR ukraine OR israel"),
    "Israel": ("he", "נתניהו OR עזה OR פוליטיקה OR ביטחון"),
    "China": ("zh", "政治 OR 经济 OR 台湾 OR 美国"),
    "Russia": ("ru", "политика OR экономика OR Украина OR США"),
    "Nigeria": ("en", "election OR Buhari OR economy OR conflict"),
    "South Africa": ("en", "Ramaphosa OR politics OR economy OR BRICS"),
    "Egypt": ("ar", "السيسي OR سياسة OR اقتصاد OR غزة"),
    "Brazil": ("pt", "política OR economia OR Lula OR Bolsonaro"),
    "India": ("en", "Modi OR BJP OR Kashmir OR economy"),
    "Pakistan": ("en", "Imran Khan OR politics OR military OR Kashmir"),
    "Iran": ("fa", "رئیسی OR سیاست OR اقتصاد OR اسرائیل"),
    "Saudi Arabia": ("ar", "بن سلمان OR سياسة OR اقتصاد OR نفط")
}


def run_country_specific_scraping():
    for country, (lang, query) in TOPICS_BY_COUNTRY.items():
        print(f"\n🌍 Récupération des articles pour {country} [{lang}] avec les mots-clés : {query}")
        articles = get_newsapi_articles(country=country, language=lang, query=query, max_pages=5)
        save_articles_to_mongo(articles)


def run_general_theme_scraping():
    for theme in THEMES:
        for lang in LANGUAGES:
            print(f"\n📰 Recherche générale : thème '{theme}' en [{lang}]")
            articles = get_newsapi_articles(country="", language=lang, query=theme, max_pages=1)
            save_articles_to_mongo(articles)


def get_articles_dataframe():
    cursor = collection.find({"emotion": {"$exists": True}})
    articles = list(cursor)
    df = pd.DataFrame(articles)
    df = df[["pays", "emotion", "score_emotion", "tags", "timestamp"]].copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    df["year"] = df["date"].dt.year
    df["nb_articles"] = 1
    df.rename(columns={"pays": "country", "tags": "theme"}, inplace=True)
    print(df.groupby("country")["emotion"].value_counts())
    print(df.groupby("year")["emotion"].value_counts())
    df = pd.DataFrame(list(collection.find({"emotion": {"$exists": True}})))
    # Assurez-vous que "pays" est renommé en "country"
    if "pays" in df.columns:
        df.rename(columns={"pays": "country"}, inplace=True)
    return df


def parse_timestamp(ts):
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        return datetime.fromisoformat(ts).isoformat()
    except:
        return datetime.now().isoformat()


def get_cleaned_articles():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["articles_db"]
    collection = db["articles"]
    migrate_country_to_pays()  # Migration des pays

    cleaned_data = []
    for article in collection.find({"emotion": {"$exists": True, "$ne": None}}):
        cleaned_article = {
            "_id": str(article["_id"]),
            "country": article.get("pays", "Unknown"),
            "emotion": article.get("emotion", "neutral"),
            "score_emotion": float(article.get("score_emotion", 0)),
            "theme": article.get("theme", "unknown"),
            "timestamp": parse_timestamp(article.get("timestamp")),
            "date": str(article.get("date", datetime.now().date())),
            "nb_articles": 1
        }
        cleaned_data.append(cleaned_article)
    if not cleaned_data:
        raise ValueError("Aucune donnée nettoyée trouvée dans MongoDB.")
    return cleaned_data


def load_emotion_model():
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle d'émotion : {e}")
        return None


def analyze_emotions(df, emotion_analyzer):
    df = df.copy()
    if "emotion" not in df.columns:
        if "content" in df.columns:
            df["emotion"] = df["content"].apply(
                lambda text: emotion_analyzer(text[:512])[0]["label"] if isinstance(text, str) and text else "unknown"
            )
        else:
            df["emotion"] = "unknown"
    df["nb_articles"] = 1
    return df


import plotly.express as px

# Mapping couleur par émotion
EMOTION_COLOR_MAP = {
    "joy": "yellow",
    "sadness": "blue",
    "anger": "red",
    "fear": "purple",
    "surprise": "orange",
    "love": "pink",
    "neutral": "grey",
}

import plotly.express as px
import pycountry


def get_iso_alpha3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None


# graphique 1
def bubble_map(df, selected_theme, selected_emotion, year=None):
    print(f"[DEBUG] Arguments reçus : year={year}, emotion={selected_emotion}, theme={selected_theme}")

    # Filtrer les données
    filtered = df[
        (df["theme"] == selected_theme) &
        (df["emotion"] == selected_emotion) &
        (df["year"] == year)
        ]

    if filtered.empty:
        raise ValueError(f"Aucune donnée pour '{selected_theme}' / '{selected_emotion}' en {year}")

    # Compter les articles par pays
    summary = filtered.groupby("country").size().reset_index(name="count")

    # Ajouter pourcentage
    total_count = summary["count"].sum()
    summary["percentage"] = (summary["count"] / total_count) * 100

    # Convertir en codes ISO alpha-3 (nécessaire pour scatter_geo)
    summary["iso_alpha"] = summary["country"].apply(get_iso_alpha3)
    summary = summary.dropna(subset=["iso_alpha"])  # Retirer les pays non reconnus

    # Couleur
    color = EMOTION_COLOR_MAP.get(selected_emotion, "grey")
    print("Données utilisées pour la carte :", summary)

    # Carte
    fig = px.scatter_geo(
        summary,
        locations="iso_alpha",
        color_discrete_sequence=[color],
        size="count",
        hover_name="country",
        hover_data={"count": True, "percentage": ':.2f'},
        projection="natural earth",
        title=f"Articles '{selected_emotion}' dans '{selected_theme}' ({year})"
    )

    fig.update_geos(
        showcountries=True,
        showland=True,
        landcolor="LightGrey",
        oceancolor="LightBlue",
        showocean=True,
        lakecolor="LightBlue"
    )

    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        geo=dict(showframe=False, showcoastlines=True)
    )

    return fig


def contour_plot(df, countries, selected_theme, year_range):
    filtered = df[(df["theme"] == selected_theme) &
                  (df["country"].isin(countries)) &
                  (df["year"].between(*year_range))]
    grouped = filtered.groupby(["year", "country", "emotion"]).size().reset_index(name="count")
    pivot = grouped.pivot_table(index="year", columns=["country", "emotion"], values="count", fill_value=0)
    z = pivot.values
    x = pivot.index.values
    y = [f"{col[0]}-{col[1]}" for col in pivot.columns]
    fig = go.Figure(data=go.Contour(z=z, x=x, y=y, colorscale="Viridis", contours_coloring='heatmap'))
    fig.update_layout(title="Évolution des émotions par pays et par année",
                      xaxis_title="Année", yaxis_title="Pays-Émotion")
    return fig


# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les articles et construire le DataFrame
    articles = get_cleaned_articles()
    df_raw = pd.DataFrame(articles)

    # Charger le modèle et annoter les émotions
    emotion_pipeline = load_emotion_model()
    df_raw = analyze_emotions(df_raw, emotion_pipeline)

    # 🔍 DEBUG : Vérification des colonnes
    print("[DEBUG] Colonnes disponibles :", df_raw.columns.tolist())

    # Ajouter colonne 'year' si absente
    if "year" not in df_raw.columns:
        df_raw["year"] = pd.to_datetime(df_raw["publishedAt"], errors="coerce").dt.year

    # 🔍 DEBUG : Aperçu des valeurs uniques
    print("[DEBUG] Valeurs uniques - année :", df_raw["year"].unique())
    print("[DEBUG] Valeurs uniques - thème :", df_raw.get("theme", pd.Series()).unique())
    print("[DEBUG] Valeurs uniques - émotion :", df_raw.get("emotion", pd.Series()).unique())

    print("Thèmes disponibles :", df_raw["theme"].unique())
    print("Émotions disponibles :", df_raw["emotion"].unique())
    print("Années disponibles :", df_raw["year"].unique())

    # Générer la carte avec paramètres fixes ou dynamiques
    fig = bubble_map(df_raw, selected_theme="politique", selected_emotion="sadness", year=2023)

    # Afficher si une figure est bien retournée
    if fig:
        fig.show()
    else:
        print("❌ Aucun graphique généré (fig est None)")

