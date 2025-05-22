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
warnings.filterwarnings("ignore")
import re
from nltk.corpus import stopwords

# === CONFIGURATION ===
NEWSAPI_KEY1 = "4aa659261bd2443997ffc38ef5c851fd"
NEWSAPI_URL1 = "https://newsapi.org/v2/everything"

NEWSAPI_URL2 = "https://newsapi.io/api/v4/search"
NEWSAPI_KEY2 = "e275d891ea6264bcdc91bfd73567c65b"
USE_CUSTOM_MODEL = False  # Toggle modèle personnalisé

# === CHARGEMENT DES PIPELINES ===
emotion_pipeline =  pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

try:
    custom_model = pipeline("text-classification", model="./emotion_model", tokenizer="./emotion_model")
except:
    custom_model = None

# === OUTILS NLP pour le nettoyage ===
tokenizer_cleaner = AutoTokenizer.from_pretrained("camembert-base")

stop_words = set(stopwords.words('french'))  # adapter à la langue

def clean_text(text):
    # passage en minuscules
    text = text.lower()
    # suppression ponctuation et caractères non alphabétiques
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text)
    # tokenization simple par espaces
    tokens = text.split()
    # suppression stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # tu peux aussi utiliser tokenizer_cleaner.tokenize si tu veux
    return " ".join(tokens)

# === MONGODB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["articles_db"]
collection = db["articles"]

# === TRADUCTION ===
def translate_to_english(text, source_lang):
    if not text or source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except:
        return text  # fallback: return as is

# Récupérer les articles depuis NewsAPI 1
def get_newsapi_articles(country, language, query="", max_pages=5):
    print(f"\n📰 NewsAPI - Articles pour {country} ({language})")

    total_articles = 0  # <- Il faut initialiser cette variable avant la boucle
    params = {
        "q": query,
        "language": language,
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY1
    }
    response = requests.get(NEWSAPI_URL1, params=params)
    if response.status_code != 200:
        print("Erreur :", response.json())
        return []
    return [
        {
            "title": a["title"],
            "author": a.get("author", "Inconnu"),
            "publication_date": a["publishedAt"],
            "content": a.get("content", ""),
            "url": a["url"],
            "source": a["source"]["name"],
            "tags": query,
            "lang": language
        } for a in response.json().get("articles", [])
    ]


# Récupérer les articles depuis NewsAPI 2 en paginant
def get_newsapi_articles(country, language, query="", max_pages=5):
    print(f"\n📰 NewsAPI - Articles pour {country} ({language})")

    total_articles = 0  # <- Il faut initialiser cette variable avant la boucle
    params = {
        "q": query,
        "language": language,
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY2
    }
    for page in range(1, max_pages + 1):
        params["page"] = page
        response = requests.get(NEWSAPI_URL2, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if not articles:
                break

            for article in articles:
                print(f"📌 {article['title']} ({article['source']['name']})")
                print(f"🔗 {article['url']}\n")
                total_articles += 1
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            break

    if total_articles == 0:
        print("❌ Aucun article trouvé.")


def save_articles_to_mongo(articles):
    count = 0
    for a in articles:
        if not collection.find_one({"url": a["url"]}):
            collection.insert_one(a)
            count += 1
    print(f"{count} article(s) ajouté(s) dans MongoDB.")

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
            # fallback sur emotion_pipeline
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
    train_dataset = train_dataset.map(lambda b: tokenizer(b["cleaned_text"], padding=True, truncation=True), batched=True)
    val_dataset = val_dataset.map(lambda b: tokenizer(b["cleaned_text"], padding=True, truncation=True), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))
    args = TrainingArguments(output_dir="./emotion_model", num_train_epochs=3, per_device_train_batch_size=4)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained("./emotion_model")
    tokenizer.save_pretrained("./emotion_model")

# === VISUALISATION ===
def plot_emotions():
    emotions = [a["emotion"] for a in collection.find({"emotion": {"$exists": True}})]
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
# === SCRIPT PRINCIPAL ===

from config import THEMES, LANGUAGES, TOPICS_BY_COUNTRY

def run_country_specific_scraping():
    for country, (lang, query) in TOPICS_BY_COUNTRY.items():
        print(f"\n🌍 Récupération des articles pour {country} [{lang}] avec les mots-clés : {query}")
        articles = get_articles(query=query, language=lang, page_size=5)
        save_articles_to_mongo(articles)

def run_general_theme_scraping():
    for theme in THEMES:
        for lang in LANGUAGES:
            print(f"\n📰 Recherche générale : thème '{theme}' en [{lang}]")
            articles = get_articles(query=theme, language=lang, page_size=5)
            save_articles_to_mongo(articles)

if __name__ == "__main__":
    print("=== Scraping par pays ciblés ===")
    run_country_specific_scraping()

    print("\n=== Scraping général multi-thèmes / multi-langues ===")
    run_general_theme_scraping()

    annotate_articles_with_emotions()
    plot_emotions()

