from pymongo import MongoClient
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import langdetect.lang_detect_exception
import re

# Fix pour résultats stables de langdetect
DetectorFactory.seed = 0

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["articles_db"]
collection = db["articles"]

# Dictionnaire principal des thèmes généraux
GENERAL_THEMES_KEYWORDS = {
    "politique": ["élection", "président", "gouvernement", "loi", "parlement", "sénat", "politicien",
                  "politics", "president", "government", "minister", "senate", "political", "réforme", "vote",
                  "Bundestag", "ministre", "réformes", "policy", "debate", "campaign", "diplomacy", "cabinet",
                  "executive", "legislation"],
    "Ukraine": ["Ukraine", "Zelensky", "Kyiv", "Kiev", "Donbass", "Crimée", "Volodymyr", "Russie", "Maidan",
                "Dnipro", "Kharkiv", "ukrainien", "war in Ukraine"],
    "Guerre": ["guerre", "conflit", "armée", "soldats", "armes", "bombes", "militaire", "invasion", "attaque",
               "war", "military", "conflict", "missile", "tanks", "frontline", "combat", "raid", "drone"],
    "culture": ["musique", "cinéma", "film", "théâtre", "littérature", "art", "peinture", "exposition",
                "spectacle", "culture", "book", "movie", "music", "film festival", "cultural", "opera",
                "concert", "sculpture", "exhibit", "gallery", "museum", "theater", "play", "novel", "literary",
                "drama", "histoire", "pape", "conclave", "vatican", "auteur", "littérature"],
    "économie": ["économie", "croissance", "PIB", "inflation", "chômage", "emploi", "revenus", "marché",
                 "fiscalité", "banque", "finance", "entreprise", "bourse", "économique", "economic",
                 "stock market", "investment", "bank", "recession", "GDP", "job market", "crise", "budget",
                 "debt", "deficit", "tariff", "inflation rate", "cost of living", "jobless", "interest rates"],
    "Israël": ["Israël", "Tel Aviv", "Hamas", "Gaza", "Palestine", "Netanyahu", "Jerusalem", "Tsahal",
               "occupation", "colonies", "israélien", "conflict in Gaza", "IDF", "Palestinians",
               "two-state solution"],
    "santé": ["covid", "pandémie", "vaccin", "hopital", "maladie", "health", "disease", "hospital",
              "virus", "WHO", "médecin", "patients", "epidemic"],
    "technologie": ["IA", "intelligence artificielle", "chatgpt", "startup", "cybersécurité", "hacking",
                    "technologie", "software", "AI", "cyberattack", "algorithme", "mongodb", "nosql",
                    "base de données", "cybersécurité", "intelligence artificielle", "algorithme", "données"],
    "environnement": ["écologie", "environnement", "pollution", "climat", "CO2", "biodiversité",
                      "déforestation", "canicule", "tempête", "climate change", "wildfire", "green energy"],
    "éducation": ["école", "étudiants", "université", "baccalauréat", "enseignement", "education",
                  "student", "university", "school", "teacher", "classroom"],
    "sport": ["football", "JO", "olympiques", "match", "score", "championnat", "soccer", "athlete",
              "basketball", "tournament", "goal", "fifa"],
    "justice": ["procès", "avocat", "jugement", "tribunal", "justice", "crime", "judge", "court",
                "lawyer", "lawsuit", "verdict", "juridique"],
    "sécurité": ["attentat", "terrorisme", "sécurité", "police", "surveillance", "terrorist", "attack",
                 "security", "cybersecurity", "radicalization"],
    "société": ["égalité", "droits", "minorités", "religion", "discrimination", "society", "social justice",
                "migration", "integration", "human rights"],
    "sciences": ["découverte", "expérience", "espace", "laboratoire", "science", "research", "nasa",
                 "scientific", "biology", "physics", "technology"],
    "transport": ["avion", "accident", "train", "bus", "trafic", "infrastructure", "transport", "flight",
                  "crash", "railway", "highway", "airport"]
}

# Mots-clés techniques spécifiques (niveau 1)
TECHNICAL_THEMES_KEYWORDS = {
    "cpp": ["cpp", "c++", "compile", "debug", "g++", "clang", "run"],
    "java": ["java", "jvm", "gradle", "maven", "jar"],
    "ai": ["machine learning", "deep learning", "neural network", "tensorflow", "pytorch", "AI"],
    "web": ["javascript", "html", "css", "react", "vue", "frontend", "web"],
    "mobile": ["android", "ios", "flutter", "react native"],
    "devops": ["docker", "kubernetes", "ci/cd", "jenkins", "ansible"],
}

# Mapping sous-thèmes -> thème général (niveau 2)
TECH_TO_GENERAL_THEME = {
    "cpp": "programmation",
    "java": "programmation",
    "ai": "intelligence artificielle",
    "web": "développement web",
    "mobile": "développement mobile",
    "devops": "infrastructure",
}


def detect_themes(text):
    if not text:
        return ["autre"]

    text = text.lower()
    detected_themes = set()

    # Détection thèmes généraux
    for theme, keywords in GENERAL_THEMES_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
                detected_themes.add(theme)
                break

    # Détection thèmes techniques (niveau 1)
    detected_technical = set()
    for theme, keywords in TECHNICAL_THEMES_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
                detected_technical.add(theme)
                break

    # Remplacer thèmes techniques par thèmes généraux associés
    for tech_theme in detected_technical:
        general = TECH_TO_GENERAL_THEME.get(tech_theme)
        if general:
            detected_themes.add(general)

    if detected_themes:
        return list(detected_themes)
    else:
        return ["autre"]


def translate_text(text, source_lang=None, target_lang="en"):
    try:
        # GoogleTranslator auto-detecte si source_lang None
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated
    except Exception as e:
        print(f"Erreur traduction: {e}")
        return text  # Retourne le texte original en cas d’erreur

def update_themes_in_db():
    count = 0
    for article in collection.find({"themes": ["autre"]}):
        content = article.get("content", "")
        if not content:
            continue
        try:
            lang = detect(content)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "unknown"

        translated_content = translate_text(content, source_lang=lang)
        themes = detect_themes(translated_content)
        print(f"Article ID: {article['_id']} | Langue détectée: {lang} => Thèmes: {themes}")

        collection.update_one({"_id": article["_id"]}, {"$set": {"themes": themes}})
        count += 1

    print(f"\nTotal d'articles traités : {count}")

    # Deuxième passe
    print("\n🔁 Ré-analyse des articles classés 'autre' avec les nouveaux mots-clés...\n")

    count_updated = 0
    for article in collection.find({"themes": ["autre"]}):
        content = article.get("content", "")
        if not content:
            continue

        try:
            lang = detect(content)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "unknown"

        translated_content = translate_text(content, source_lang=lang)
        themes = detect_themes(translated_content)

        if themes != ["autre"]:
            collection.update_one({"_id": article["_id"]}, {"$set": {"themes": themes}})
            print(f"🔁 Article ID: {article['_id']} | Nouveau thème détecté: {themes}")
            count_updated += 1
        else:
            print(f"🔍 ARTICLE {article['_id']} en {lang} | Texte traduit :\n{translated_content}\n---")

    print(f"\n✅ Articles reclassés : {count_updated}")
