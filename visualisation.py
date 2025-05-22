import pymongo
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
from dash import Dash, dcc, html, Input, Output


# --- Connexion MongoDB ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["articles_db"]
collection = db["articles"]

def create_dash_app(flask_app):
    documents = list(collection.find({"emotion": {"$exists": True}}))
    print(f"Nombre d'articles avec émotion : {len(documents)}")

    data = []
    for doc in documents:
        emotion = doc.get("emotion")
        country = doc.get("country", "Inconnu")

        theme = doc.get("tags", "Autre")
        date_str = doc.get("publication_date")
        try:
            year = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").year
        except:
            year = None

        if emotion and theme:
            data.append({
                "theme": theme,
                "country": country,
                "emotion": emotion,
                "year": year,
                "count": 1
            })

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("Aucune donnée à afficher. Vérifie que la base contient des émotions annotées.")

    df_grouped = df.groupby(["theme", "country", "emotion", "year"]).sum().reset_index()

    emotion_colors = {
        "joy": "yellow",
        "sadness": "blue",
        "anger": "red",
        "fear": "purple",
        "surprise": "orange",
        "love": "pink",
        "neutral": "grey"
    }

    dash_app.layout = html.Div([
        html.H2("Cartographie des émotions par thème, émotion et pays"),

        html.Label("Thèmes :"),
        dcc.Dropdown(
            id='theme-dropdown',
            options=[{"label": t, "value": t} for t in df_grouped["theme"].unique()],
            value=df_grouped["theme"].unique()[0]
        ),

        html.Label("Émotion :"),
        dcc.Dropdown(
            id='emotion-dropdown',
            options=[
                {"label": emo.capitalize(), "value": emo}
                for emo in df_grouped["emotion"].unique()
            ],
            value=df_grouped["emotion"].unique()[0]
        ),

        html.Label("Année :"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[
                {"label": str(y), "value": y}
                for y in sorted(df_grouped["year"].dropna().unique())
            ],
            value=sorted(df_grouped["year"].dropna().unique())[0]
        ),

        dcc.Graph(id='bubble-chart')
    ])

    @dash_app.callback(
        Output("bubble-chart", "figure"),
        Input("theme-dropdown", "value"),
        Input("emotion-dropdown", "value"),
        Input("year-dropdown", "value"),
    )
    def update_bubble(theme, emotion, year):
        return bubble_map(df, theme, emotion, year)

    def update_graphs(selected_theme):
        filtered = df_grouped[df_grouped["theme"] == selected_theme]
        print("Data pour le thème sélectionné :", filtered.head())

        fig_bar = px.bar(
            filtered,
            x="country",
            y="count",
            color="emotion",
            color_discrete_map=emotion_colors,
            title=f"Nombre d’articles par émotion dans le thème : {selected_theme}"
        )

        fig_bubble = px.scatter_geo(
            filtered,
            locations="country",
            locationmode="country names",
            size="count",
            color="emotion",
            projection="natural earth",
            hover_name="country",
            title=f"Carte des émotions (bubble) : {selected_theme}",
            color_discrete_map=emotion_colors
        )

        return fig_bar, fig_bubble

    return dash_app
