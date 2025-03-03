
import streamlit as st
import pandas as pd
import networkx as nx
import hvplot.pandas

import plotly.express as px
import datasets
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.manifold import TSNE
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


import hvplot.pandas
import panel as pn

# Sidebar for navigation between different pages
page = st.sidebar.radio("Select a page", ("Home", "TF-IDF","Word Embeddings","Tweet Embeddings", "Prediction model"))


if page == "Home":
    # Your original page content
    st.title("Welcome to the Event Dashboard")
    st.write("Here you can select events and view the corresponding data.")
    # Add your existing code here (e.g., event selection, buttons, etc.)

    # Chemin du fichier GraphML
    path_data = '/Users/sarahbouchet/Desktop/M2_TSE/S2/Web_mining/Project_documents/database_formated_for_NetworkX.graphml'

    # Chargement du graphe
    graph = nx.read_graphml(path_data)

    # Extraction des labels uniques
    unique_labels = set(data.get("labels") for _, data in graph.nodes(data=True))

    # Créer un dictionnaire pour stocker la correspondance entre "topic" et "inter" du nœud "Event"
    topic_to_inter = {}

    # Créer un ensemble pour les topics uniques
    unique_topics = set()

    # Parcourir tous les nœuds de type "Tweet" pour extraire les topics uniques
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            # Extraire le topic du tweet
            topic = tweet_data.get("topic")  # Remplacer "topic" par l'attribut réel
            if topic:  # Vérifie si un topic est présent
                unique_topics.add(topic)

    # Pour chaque topic unique, associer l'événement et son attribut "inter"
    for topic in unique_topics:
        # Parcourir les nœuds de type "Event" pour associer le topic à l'événement correspondant
        for event_node, event_data in graph.nodes(data=True):
            if event_data.get("labels") == ":Event" and topic in event_data.get("trecisid", ""):
                # Extraire l'attribut "inter" du nœud "Event"
                id = event_data.get("id")
                # Ajouter la correspondance dans le dictionnaire
                topic_to_inter[topic] = id

    # Créer un dictionnaire pour stocker la correspondance entre Tweet et User
    tweet_to_user = {}

    # Parcourir tous les edges pour trouver ceux de type "POSTED"
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("label") == "POSTED":
            # Trouver le User (u) et le Tweet (v)
            user_id = graph.nodes[u].get("id")  # id du User
            tweet_id = graph.nodes[v].get("id")  # id du Tweet

            # Associer le Tweet à son User
            tweet_to_user[tweet_id] = user_id

    # Dictionnaire pour associer les topics des tweets à leur EventType
    topic_to_event_type = {}

    # Parcourir tous les nœuds de type 'Event' pour construire la correspondance topic -> EventType
    for event_node, event_data in graph.nodes(data=True):
        if event_data.get("labels") == ":Event":  # On vérifie que c'est un nœud de type Event
            event_id = event_data.get("trecisid")  # Le 'id' de l'événement (correspond au 'topic' dans le tweet)
            event_type = event_data.get("eventType")  # L'EventType explicite de l'événement
            if event_id and event_type:
                topic_to_event_type[event_id] = event_type  # Associer le 'topic' (id de l'événement) à 'eventType'

    # Pour chaque tweet, associer son EventType en fonction du topic
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":  # Vérifier que c'est un nœud de type Tweet
            tweet_topic = tweet_data.get("topic")  # Le topic du tweet, à relier à un événement
            if tweet_topic and tweet_topic in topic_to_event_type:
                event_type_for_tweet = topic_to_event_type[tweet_topic]  # Récupérer l'EventType associé au topic
                tweet_data['eventType'] = event_type_for_tweet  # Ajouter l'EventType au tweet

    # Dictionnaire pour compter les tweets par type d'événement
    event_type_count = {}

    # Parcourir tous les nœuds de type 'Tweet' pour compter l'effectif par type d'événement
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet" and 'eventType' in tweet_data:
            event_type = tweet_data['eventType']  # Le type d'événement associé au tweet

            # Compter le nombre de tweets pour chaque type d'événement
            if event_type:
                if event_type not in event_type_count:
                    event_type_count[event_type] = 0
                event_type_count[event_type] += 1



    # Liste des types d'événements
    event_types = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']

    # Stocker les données
    data = []

    # Parcourir tous les types d'événements
    for event_type in event_types:
        tweet_dates = []

        # Parcourir les nœuds de type 'Tweet' et associer les tweets à l'événement spécifié
        for tweet_node, tweet_data in graph.nodes(data=True):
            if tweet_data.get("labels") == ":Tweet" and tweet_data.get("eventType") == event_type:
                if 'created_at' in tweet_data:
                    tweet_time = tweet_data['created_at']
                    tweet_dates.append(tweet_time)  # Ajouter la date du tweet

        # Convertir les dates en format datetime
        tweet_dates = pd.to_datetime(tweet_dates, errors='coerce')
        tweet_dates = tweet_dates.dropna()  # Supprimer les valeurs NaT

        # Compter le nombre de tweets par date
        # Convert tweet_dates to a Series before using groupby and size
        tweet_dates_series = pd.Series(tweet_dates)
        tweet_counts = tweet_dates_series.groupby(tweet_dates_series.dt.date).size()

        # Ajouter les données à la table
        for date, count in tweet_counts.items():
            data.append({'date': date, 'Event_Type': event_type, 'num_tweets_perday': count})

    # Création du DataFrame
    df_tweets_time_series = pd.DataFrame(data)

    # Vérification des premières lignes
    (df_tweets_time_series.head())


    import pandas as pd

    # Liste des types d'événements et leurs couleurs
    event_types = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']
    event_colors = {
        'typhoon': 'royalblue',
        'shooting': 'darkorange',
        'wildfire': 'green',
        'bombing': 'red',
        'earthquake': 'purple',
        'flood': 'cyan'
    }

    # Liste pour stocker les statistiques pour chaque type d'événement
    event_stats = []

    # Liste pour stocker les données des relations IS_ABOUT
    data_is_about = []

    # Parcourir les relations IS_ABOUT (user - event)
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("label") == "IS_ABOUT":  # Vérifier le bon label
            data_is_about.append({"user": u, "event": v, "label": "IS_ABOUT"})

    # Créer un DataFrame des relations IS_ABOUT
    df_is_about = pd.DataFrame(data_is_about)

    # Liste pour stocker les données des événements
    event_data = []

    # Parcourir les nœuds pour trouver ceux de type "Event"
    for node, data in graph.nodes(data=True):
        if data.get("labels") == ":Event":  # Vérifie si c'est un Event
            event_data.append({"event": node, "event_id": data.get("id"), "eventType": data.get("eventType")})

    # Créer le DataFrame des événements
    df_events = pd.DataFrame(event_data)

    # Joindre les DataFrames sur la colonne "event"
    df_combined = pd.merge(df_is_about, df_events, on='event', how='left')

    # Regrouper par "eventType" et compter le nombre unique d'utilisateurs
    user_counts_by_event = df_combined.groupby('eventType')['user'].nunique().reset_index()
    sous_event_id_counts = df_combined.groupby(['eventType'])['event_id'].nunique().reset_index(name='unique_event_id_count')

    # Renommer la colonne pour correspondre à l'output attendu
    user_counts_by_event = user_counts_by_event.rename(columns={"user": "Number of Users"})
    sous_event_id_counts = sous_event_id_counts.rename(columns={"user": "Number of unique event in the category"})

    # Afficher les résultats


    # Calcul des statistiques pour chaque type d'événement
    for event_type in event_types:
        tweet_dates = []
        user_ids = []

        # Sélectionner les utilisateurs et tweets associés à cet événement
        event_df = df_combined[df_combined['eventType'] == event_type]

        # Parcourir les nœuds de type 'Tweet' et associer les tweets avec l'événement
        for tweet_node, tweet_data in graph.nodes(data=True):
            if tweet_data.get("labels") == ":Tweet" and tweet_data.get("eventType") == event_type:
                if 'created_at' in tweet_data:
                    tweet_time = tweet_data['created_at']  # Récupérer le timestamp du tweet
                    user_id = tweet_data['id']  # ID de l'utilisateur
                    tweet_dates.append(tweet_time)  # Ajouter la date du tweet à la liste
                    user_ids.append(user_id)  # Ajouter l'ID de l'utilisateur à la liste

        # Convertir les dates des tweets en datetime
        tweet_dates = pd.to_datetime(tweet_dates, errors='coerce')

        # Supprimer les valeurs invalides (NaT)
        tweet_dates = tweet_dates.dropna()

        # Créer une série des dates des tweets
        tweet_dates_series = pd.Series(tweet_dates)

        # Compter le nombre de tweets par jour
        tweet_counts = tweet_dates_series.groupby(tweet_dates_series.dt.date).size()

        # Calcul des statistiques pour l'événement
        num_tweets = tweet_counts.sum()  # Nombre total de tweets pour cet événement
        num_users = user_counts_by_event[user_counts_by_event['eventType'] == event_type]['Number of Users'].values[0]  # Nombre d'utilisateurs uniques
        num_subevent = sous_event_id_counts[sous_event_id_counts['eventType'] == event_type]['unique_event_id_count'].values[0]
        first_tweet_date = tweet_dates.min()  # Date du premier tweet
        last_tweet_date = tweet_dates.max()  # Date du dernier tweet
        avg_tweet_freq = (last_tweet_date - first_tweet_date).days / num_tweets if num_tweets > 0 else 0  # Fréquence moyenne des tweets en jours


        # Ajouter les statistiques à la liste
        event_stats.append({
            'Event_Type': event_type,
            'Number_of_Tweets_perEvent': num_tweets,
            'Number_of_Users_perEvent': num_users,
            'Nb_of_sub_event': num_subevent,
            'First_Tweet_Date': first_tweet_date,
            'Last_Tweet_Date': last_tweet_date,
            'Avg_Tweet_Frequency': avg_tweet_freq
        })

    # Créer un DataFrame à partir des statistiques
    event_stats_df = pd.DataFrame(event_stats)
    df_complete = df_tweets_time_series.merge(event_stats_df, on=['Event_Type'], how='left')

    # Regrouper par 'Event_Type' et calculer les statistiques
    df_resume = df_complete.groupby('Event_Type').agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max'),  # Total des tweets par événement
        Number_of_Users_perEvent=('Number_of_Users_perEvent', 'max'),  # Nombre unique d'utilisateurs
        Nb_of_sub_event=('Nb_of_sub_event', 'max'),  # Nombre total de sous-événements
        First_Tweet_Date=('date', 'min'),  # Date du premier tweet
        Last_Tweet_Date=('date', 'max'),  # Date du dernier tweet
        Avg_Tweet_Frequency=('Avg_Tweet_Frequency', 'max')  # Fréquence moyenne des tweets
    ).reset_index()


    # Afficher le graphique
    # Add Streamlit title
    st.title("Temporal distribution of tweets for each type of event")



    st.sidebar.write('Select Filter')

    # Assume df_complete is already loaded
    choices = list(df_complete['Event_Type'].unique())  # Get unique event types

    # Assume df_complete is already loaded
    choices = list(df_complete['Event_Type'].unique())  # Get unique event types

    # Initialize session state for button states (if not already initialized)
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = ['typhoon']  #pre selection to see something in the app

    col_count = len(choices)
    columns = st.columns(col_count)  # Create as many columns as the number of events

    # A button for each event type in the columns
    for idx, event in enumerate(choices):
        # Check if the event is selected (based on session state)
        is_selected = event in st.session_state.selected_events
        
        # Change the button style to grey if selected
        button_color = 'background-color: gray;' if is_selected else 'background-color: lightblue;'
        
        # Create a button in the respective column
        with columns[idx]:
            if st.button(f'{event}', key=event, help=f'Select {event}', use_container_width=False):
                if is_selected:
                    st.session_state.selected_events.remove(event)  # Remove if already selected
                else:
                    st.session_state.selected_events.append(event)  # Add to selected events list

    # Display selected events
    if st.session_state.selected_events:
        st.write(f"### Data for selected events: {', '.join(st.session_state.selected_events)}")
        df_resume = df_resume[df_resume['Event_Type'].isin(st.session_state.selected_events)]
        st.dataframe(df_resume)
    else:
        st.write("Please select at least one event type")

    # # Assurez-vous que df_complete est chargé
    # choices = list(df_complete['Event_Type'].unique())  # Convertir en liste pour la sélection
    # selected_choices = st.selectbox('Select an event type', choices, index=len(choices)-1)

    # df_resume = df_resume[df_resume["Event_Type"] == selected_choices]
    # st.dataframe(df_resume)


    # Assurez-vous que la colonne 'date' est bien au format datetime
    df_complete['date'] = pd.to_datetime(df_complete['date'])
    # filter data based on selection
    df_flt = df_complete[df_complete['Event_Type'].isin(st.session_state.selected_events)]

    df_flt = df_flt.groupby(['Event_Type', 'date']).agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max')  # Max number of tweets per day per event
    ).reset_index()

    # Création du graphique
    fig = px.line(df_flt, 
                x='date', 
                y='Number_of_Tweets_per_day', 
                color='Event_Type', 
                title="Evolution of the number of tweets posted each day given the event type",
                labels={'num_tweets_perday': 'Nombre de Tweets par Jour', 'date': 'Date'},
                line_shape='linear',  # Pour des courbes lissées
                markers=True,  # Ajoute des points sur les courbes
                template="plotly_white")  # Style professionnel

    # Personnalisation de l'apparence
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of tweets posted",
        legend_title="Event type",
        hovermode="x unified",  # Améliore l'affichage des valeurs au survol
        font=dict(family="Arial, sans-serif", size=14),  # Police plus lisible
    )

    st.plotly_chart(fig)

                                
    if df_flt.shape[0]>0:
        st.dataframe(df_flt)
    else:
        st.write("Empty Dataframe")

### to add : duration of the tweet period
### panel for the three embedings 

elif page == "TF-IDF":
    # This page displays "Hello"
    st.title("Hello Page")
    st.write("Hello! Welcome to this page.")

elif page == "Word Embeddings":
    # This page displays "Hello"
    st.title("Hello Page")
    st.write("Hello! Welcome to this page.")

elif page == "Tweet Embeddings":

    def get_word_embeddings():
        # Download and load a pre-trained Word2Vec model
        nltk.download('punkt_tab')
        model = api.load("word2vec-google-news-300")
        return model

    def get_tweet_word_embedding(doc, model):
        def preprocess_text(text):
            return word_tokenize(text.lower())

        words = preprocess_text(doc)
        word_vectors = []
        for word in words:
            if word in model:
                word_vectors.append(model[word])

        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)

        document_embedding = np.mean(word_vectors, axis=0)
        return document_embedding
    w2v_model = get_word_embeddings()

    # This function converts each tweet to a vector :

    tweet_embeddings = []

    # Iterate over the nodes of type 'Tweet'
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet" and 'text' in tweet_data:  # Assuming 'text' key holds tweet content
            tweet_text = tweet_data['text']  # Extract the text of the tweet
            tweet_vector = get_tweet_word_embedding(tweet_text, w2v_model)  # Get the embedding
            tweet_embeddings.append(tweet_vector)  # Append the vector representation

    # You now have `tweet_embeddings` which is a list of vectors representing the tweets

    # Now, let's convert them into a DataFrame or further processing
    tweet_embeddings_df = pd.DataFrame(tweet_embeddings)  # Create DataFrame for analysis


    # Perform t-SNE to reduce dimensions to 2D
    tsne_w2v = TSNE(perplexity=15, n_components=2, init='pca', n_iter=1000, random_state=42)
    w2v_2d = tsne_w2v.fit_transform(np.array(tweet_embeddings))  # Transpose to work with word vectors

    # creation of the data : 
    event_types = []

    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet" and 'eventType' in tweet_data:
            event_type = tweet_data['eventType']  # Récupérer uniquement le type d'événement
            event_types.append(event_type)

    event_types_df = pd.DataFrame({'eventType': event_types})

    tweet_embeddings_df = pd.DataFrame(tweet_embeddings)
    tweet_embeddings_df['eventType'] = event_types

    # Créer un DataFrame avec les coordonnées 2D et les types d'événements
    w2v_2d_df = pd.DataFrame(w2v_2d, columns=["x", "y"])
    w2v_2d_df["eventType"] = event_types

    # Création du widget de filtrage
    event_filter = pn.widgets.Select(name="Event Type", options=list(w2v_2d_df["eventType"].unique()))

    # Fonction de mise à jour du graphique
    @pn.depends(event_filter)
    def update_plot(eventType):
        filtered_df = w2v_2d_df[w2v_2d_df["eventType"] == eventType]
        return filtered_df.hvplot.scatter(
            x="x", y="y",
            title=f"Tweet Embeddings pour {eventType}",
            xlabel="t-SNE X", ylabel="t-SNE Y",
            size=50, hover_cols=["eventType"], color="blue"
        )

    # Création du dashboard interactif
    dashboard = pn.Column(event_filter, update_plot)

    st.title("Hello Page")
    st.write(dashboard)

elif page == "Prediction model":
    # This page displays "Hello"
    st.title("Hello Page")
    st.write("Hello! Welcome to this page.")