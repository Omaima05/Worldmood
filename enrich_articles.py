from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator
import time

NEWSAPI_URL = "https://newsapi.org/api/v4/Everything"
NEWSAPI_KEY = "5cc959f32-de5449ea982ea306e99cd0d"

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["articles_db"]
collection = db["articles"]


# Liste des pays avec mot-clé et langue
countries = {
    "USA": {"query": "United States", "language": "en"},
    "Germany": {"query": "Germany", "language": "de"},
    "Israel": {"query": "Israel", "language": "en"},
    "China": {"query": "China", "language": "zh"},
    "Russia": {"query": "Russia", "language": "ru"},
    "Nigeria": {"query": "Nigeria", "language": "en"},
    "South Africa": {"query": "South Africa", "language": "en"},
    "Egypt": {"query": "Egypt", "language": "ar"},
    "Brazil": {"query": "Brazil", "language": "pt"},
    "India": {"query": "India", "language": "en"},
    "Pakistan": {"query": "Pakistan", "language": "en"},
    "Iran": {"query": "Iran", "language": "fa"},
    "Saudi Arabia": {"query": "Saudi Arabia", "language": "ar"},
}

# User-Agent pour éviter blocage
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/114.0.0.0 Safari/537.36"
}

# Fonction détection thème simplifiée
def detect_theme(text):
    text = text.lower()
    if "climate" in text or "environment" in text:
        return "climat"
    elif "war" in text or "conflict" in text:
        return "guerre"
    elif "economy" in text or "inflation" in text:
        return "économie"
    elif "culture" in text or "art" in text:
        return "culture"
    elif "politic" in text or "election" in text or "government" in text:
        return "politique"
    return "autre"

# Fonction de scraping Bing News pour un pays donné

def scrape_bing_news(country_name, query, language):
    search_url = f"https://www.bing.com/news/search?q={query}&qft=lang:{language}&form=QBNH"
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Erreur HTTP pour {country_name}: {e}")
        return

    soup = BeautifulSoup(response.content, "html.parser")
    news_items = soup.find_all("a", {"class": "title"}, href=True)

    for link in news_items:
        title = link.get_text(strip=True)
        href = link["href"]

        if not title or len(title) < 20 or collection.find_one({"link": href}):
            continue
        # Éviter doublons
        if collection.find_one({"link": href}):
            continue

        try:
            detected_lang = detect(title)
        except Exception:
            detected_lang = language

        if detected_lang != "en":
            try:
                title_en = GoogleTranslator(source='auto', target='en').translate(title)
            except Exception:
                title_en = title
        else:
            title_en = title

        theme = detect_theme(title_en)

        doc = {
            "title_original": title,
            "title_en": title_en,
            "link": href,
            "source": "bing_news",
            "theme": theme,
            "country": country_name,
            "language": detected_lang
        }
        collection.insert_one(doc)
        print(f"[{country_name}] Ajouté: {title_en} - {theme}")

    time.sleep(2)
# Boucle principale
for country_name, info in countries.items():
    print(f"Scraping articles pour {country_name}...")
    scrape_bing_news(country_name, info["query"], info["language"])

