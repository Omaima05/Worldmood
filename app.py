from flask import Flask, render_template, send_file
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client["world_mood"]
collection = db["articles"]

@app.route("/")
def index():
    # Optionnel : (ré)exécuter les fonctions
    topics = ["politique", "écologie", "technologie"]
    langs = ["fr", "en"]
    method = "gnews"

    for topic in topics:
        for lang in langs:
            articles = fetch_articles(method=method, query=topic, lang=lang, max_articles=5)
            save_articles_to_mongo(articles)

    annotate_articles_with_emotions()
    plot_emotions()  # crée le graphique

    # On extrait les dernières émotions
    articles = list(collection.find({"emotion": {"$exists": True}}).sort("publication_date", -1).limit(20))
    return render_template("index.html", articles=articles)

@app.route("/plot.png")
def plot_png():
    return send_file("static/plot.png", mimetype='image/png')

if __name__name__ == "_final2_":
    app.run(debug=True)