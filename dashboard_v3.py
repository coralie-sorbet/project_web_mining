import os
import re
import time
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import hvplot.pandas
import streamlit as st

# --- Scikit-learn ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- NLTK ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet as wn, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.data.clear_cache()
# Define custom NLTK data path
nltk_data_path = "/tmp/nltk_data"

import os
import nltk

# Define custom NLTK data path
nltk_data_path = "/tmp/nltk_data"

# Ensure NLTK uses the correct path
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path.append(nltk_data_path)

# Download resources if not already present
resources = ["stopwords", "punkt", "wordnet", "vader_lexicon", "sentiwordnet"]

for resource in resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        try:
            nltk.download(resource, download_dir=nltk_data_path)
        except FileExistsError:
            pass  # Ignore if the directory already exists

# # Force-download 'punkt' tokenizer if not found
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt", download_dir=nltk_data_path)
# import torch  
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", download_dir=nltk_data_path)
import torch  

from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import gensim.downloader as api

@st.cache_data(show_spinner=True)
def build_tfidf(docs: list[str]):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)
    return vectorizer, tfidf

# =============================================================================
# Navigation via Streamlit
# =============================================================================
page = st.sidebar.radio("Select a page", (
    "Home",
    "TF-IDF",
    "Word Embeddings",
    "Tweet Embeddings",
    "Sentiment Analysis",
    "Search System"    
))

# Fonction utilitaire pour charger le graphe (cached)
@st.cache_data(show_spinner=True)
def load_graph(path: str) -> nx.Graph:
    if not os.path.exists(path):
        st.error(f"Graph file not found at {path}. Please check your file path.")
        return None
    try:
        return nx.read_graphml(path)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None

# =============================================================================
# PAGE "Home" : Dashboard et statistiques d'événements
# =============================================================================
if page == "Home":
    st.title("Welcome to the Event Dashboard")
    st.write("Here you can select events and view the corresponding data.")

    # Chemin du fichier GraphML
    path_data = "database/Everything/database_formated_for_NetworkX.graphml"

    # Chargement du graph via la fonction en cache
    graph = load_graph(path_data)
    if graph is None:
        st.stop()

    # Extraction des labels uniques
    unique_labels = set(data.get("labels") for _, data in graph.nodes(data=True))

    # Créer un dictionnaire pour stocker la correspondance entre "topic" et "inter" du nœud "Event"
    topic_to_inter = {}

    # Créer un ensemble pour les topics uniques
    unique_topics = set()

    # Parcourir tous les nœuds de type "Tweet" pour extraire les topics uniques
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            topic = tweet_data.get("topic")
            if topic:
                unique_topics.add(topic)

    # Pour chaque topic unique, associer l'événement et son attribut "inter"
    for topic in unique_topics:
        for event_node, event_data in graph.nodes(data=True):
            if event_data.get("labels") == ":Event" and topic in event_data.get("trecisid", ""):
                id = event_data.get("id")
                topic_to_inter[topic] = id

    # Créer un dictionnaire pour stocker la correspondance entre Tweet et User
    tweet_to_user = {}
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("label") == "POSTED":
            user_id = graph.nodes[u].get("id")
            tweet_id = graph.nodes[v].get("id")
            tweet_to_user[tweet_id] = user_id

    # Dictionnaire pour associer les topics des tweets à leur EventType
    topic_to_event_type = {}
    for event_node, event_data in graph.nodes(data=True):
        if event_data.get("labels") == ":Event":
            event_id = event_data.get("trecisid")
            event_type = event_data.get("eventType")
            if event_id and event_type:
                topic_to_event_type[event_id] = event_type

    # Pour chaque tweet, associer son EventType en fonction du topic
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            tweet_topic = tweet_data.get("topic")
            if tweet_topic and tweet_topic in topic_to_event_type:
                event_type_for_tweet = topic_to_event_type[tweet_topic]
                tweet_data['eventType'] = event_type_for_tweet

    # Dictionnaire pour compter les tweets par type d'événement
    event_type_count = {}
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet" and 'eventType' in tweet_data:
            event_type = tweet_data['eventType']
            if event_type:
                event_type_count[event_type] = event_type_count.get(event_type, 0) + 1

    # Liste des types d'événements
    event_types = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']

    # Stocker les données pour la série temporelle
    data = []
    for event_type in event_types:
        tweet_dates = []
        for tweet_node, tweet_data in graph.nodes(data=True):
            if tweet_data.get("labels") == ":Tweet" and tweet_data.get("eventType") == event_type:
                if 'created_at' in tweet_data:
                    tweet_dates.append(tweet_data['created_at'])
        # Convertir les dates en format datetime et retirer les valeurs NaT
        tweet_dates = pd.to_datetime(tweet_dates, errors='coerce').dropna()
        # Création d'une série de dates (conversion explicite en date)
        tweet_counts = pd.Series([x.date() for x in tweet_dates]).value_counts().sort_index()
        for date, count in tweet_counts.items():
            data.append({'date': date, 'Event_Type': event_type, 'num_tweets_perday': count})

    df_tweets_time_series = pd.DataFrame(data)
    df_tweets_time_series.head()

    # Liste des types d'événements et leurs couleurs (pour une éventuelle utilisation)
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
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("label") == "IS_ABOUT":
            data_is_about.append({"user": u, "event": v, "label": "IS_ABOUT"})

    df_is_about = pd.DataFrame(data_is_about)

    # Liste pour stocker les données des événements
    event_data = []
    for node, data in graph.nodes(data=True):
        if data.get("labels") == ":Event":
            event_data.append({"event": node, "event_id": data.get("id"), "eventType": data.get("eventType")})
    df_events = pd.DataFrame(event_data)

    # Joindre les DataFrames sur la colonne "event"
    df_combined = pd.merge(df_is_about, df_events, on='event', how='left')

    # Regrouper par "eventType" et compter le nombre unique d'utilisateurs
    user_counts_by_event = df_combined.groupby('eventType')['user'].nunique().reset_index()
    sous_event_id_counts = df_combined.groupby(['eventType'])['event_id'].nunique().reset_index(name='unique_event_id_count')

    user_counts_by_event = user_counts_by_event.rename(columns={"user": "Number of Users"})
    sous_event_id_counts = sous_event_id_counts.rename(columns={"user": "Number of unique event in the category"})

    # Calcul des statistiques pour chaque type d'événement
    for event_type in event_types:
        tweet_dates = []
        user_ids = []
        event_df = df_combined[df_combined['eventType'] == event_type]
        for tweet_node, tweet_data in graph.nodes(data=True):
            if tweet_data.get("labels") == ":Tweet" and tweet_data.get("eventType") == event_type:
                if 'created_at' in tweet_data:
                    tweet_dates.append(tweet_data['created_at'])
                    user_ids.append(tweet_data['id'])
        tweet_dates = pd.to_datetime(tweet_dates, errors='coerce').dropna()
        tweet_counts = pd.Series([x.date() for x in tweet_dates]).value_counts().sort_index()
        num_tweets = tweet_counts.sum()
        num_users = user_counts_by_event[user_counts_by_event['eventType'] == event_type]['Number of Users'].values[0]
        num_subevent = sous_event_id_counts[sous_event_id_counts['eventType'] == event_type]['unique_event_id_count'].values[0]
        first_tweet_date = tweet_dates.min()
        last_tweet_date = tweet_dates.max()
        avg_tweet_freq = (last_tweet_date - first_tweet_date).days / num_tweets if num_tweets > 0 else 0
        event_stats.append({
            'Event_Type': event_type,
            'Number_of_Tweets_perEvent': num_tweets,
            'Number_of_Users_perEvent': num_users,
            'Nb_of_sub_event': num_subevent,
            'First_Tweet_Date': first_tweet_date,
            'Last_Tweet_Date': last_tweet_date,
            'Avg_Tweet_Frequency': avg_tweet_freq
        })

    event_stats_df = pd.DataFrame(event_stats)
    df_complete = df_tweets_time_series.merge(event_stats_df, on=['Event_Type'], how='left')

    df_resume = df_complete.groupby('Event_Type').agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max'),
        Number_of_Users_perEvent=('Number_of_Users_perEvent', 'max'),
        Nb_of_sub_event=('Nb_of_sub_event', 'max'),
        First_Tweet_Date=('date', 'min'),
        Last_Tweet_Date=('date', 'max'),
        Avg_Tweet_Frequency=('Avg_Tweet_Frequency', 'max')
    ).reset_index()

    st.title("Temporal distribution of tweets for each type of event")
    st.sidebar.write('Select Filter')

    choices = list(df_complete['Event_Type'].unique())
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = ['typhoon']

    col_count = len(choices)
    columns = st.columns(col_count)
    for idx, event in enumerate(choices):
        is_selected = event in st.session_state.selected_events
        with columns[idx]:
            if st.button(f'{event}', key=event, help=f'Select {event}', use_container_width=False):
                if is_selected:
                    st.session_state.selected_events.remove(event)
                else:
                    st.session_state.selected_events.append(event)

    if st.session_state.selected_events:
        st.write(f"### Data for selected events: {', '.join(st.session_state.selected_events)}")
        df_resume = df_resume[df_resume['Event_Type'].isin(st.session_state.selected_events)]
        st.dataframe(df_resume)
    else:
        st.write("Please select at least one event type")

    df_complete['date'] = pd.to_datetime(df_complete['date'])
    df_flt = df_complete[df_complete['Event_Type'].isin(st.session_state.selected_events)]
    df_flt = df_flt.groupby(['Event_Type', 'date']).agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max')
    ).reset_index()

    fig = px.line(df_flt, 
                x='date', 
                y='Number_of_Tweets_per_day', 
                color='Event_Type', 
                title="Evolution of the number of tweets posted each day given the event type",
                labels={'num_tweets_perday': 'Nombre de Tweets par Jour', 'date': 'Date'},
                line_shape='linear',
                markers=True,
                template="plotly_white")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of tweets posted",
        legend_title="Event type",
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
    )
    st.plotly_chart(fig)
                                
    if df_flt.shape[0] > 0:
        st.dataframe(df_flt)
    else:
        st.write("Empty Dataframe")

  
# =============================================================================
# PAGE "TF-IDF"
# =============================================================================
elif page == "TF-IDF":
    st.title("TF-IDF Analysis")
    st.write("Here we analyse the words in tweets linked to different types of events using TF-IDF.")

    # --- Text Preprocessing ---
    def clean_tweet(tweet: str) -> str:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english")).union({"http", "https", "rt", "news", "amp", "nhttps"})
        tweet = re.sub(r"http\S+|www\S+", '', tweet)  
        tweet = re.sub(r'@\w+|#\w+', '', tweet) 
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  
        tweet = tweet.lower() 
        tokens = word_tokenize(tweet) 
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] 
        return " ".join(tokens)

    # --- Function TF-IDF representation ---
    def get_tfidf_representations(texts):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100) 
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        return tfidf_vectorizer, tfidf_matrix
    
    # --- display the top_n most frequent words  --- 
    def plot_tfidf_keywords(tfidf_df, top_n):
        top_keywords = tfidf_df.mean(axis=0).sort_values(ascending=False).head(top_n)
        df_keywords = pd.DataFrame({
            'word': top_keywords.index,
            'tfidf_score': top_keywords.values
        })
        fig = px.bar(df_keywords, 
                    x='word', 
                    y='tfidf_score', 
                    title='Top TF-IDF Keywords',
                    labels={'word': 'Words', 'tfidf_score': 'TF-IDF Score'},
                    color='tfidf_score', 
                    color_continuous_scale='Viridis')  

        fig.update_layout(xaxis_title="Words", yaxis_title="TF-IDF Score")
        st.plotly_chart(fig) 
    
    # --- Visualise words with TSNE ---
    def plot_single_points(all_words, word_to_index, vis_2d):
        data = {
            "word": all_words,
            "x": [vis_2d[word_to_index[word], 0] for word in all_words],
            "y": [vis_2d[word_to_index[word], 1] for word in all_words]}
        
        df_vis = pd.DataFrame(data)
        fig = px.scatter(
            df_vis,
            x="x",
            y="y",
            text="word",      
            title="Word Vector Representations",
            labels={"x": "t-SNE X", "y": "t-SNE Y"},
            hover_data=["word"] 
        )
        
        fig.update_traces(textposition='top center')  
        st.plotly_chart(fig) 

    # --- Cluster ---
    def cluster_keywords_by_event(event_type, tweets, K):
        tfidf_vectorizer, tfidf_matrix = get_tfidf_representations(tweets)
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(tfidf_matrix.T)

        words = tfidf_vectorizer.get_feature_names_out()
        word_clusters = {}
        for i in range(K):  # k clusters
            word_clusters[i] = [words[index] for index in range(len(words)) if kmeans.labels_[index] == i]

        return word_clusters, tfidf_vectorizer, tfidf_matrix

    # ---For displaying word clusters with PCA ---
    def plot_word_clusters_PCA(event_type, clusters):
        all_words = []
        all_labels = []

        for cluster, words in clusters.items():
            all_words.extend(words)
            all_labels.extend([cluster] * len(words))
        
        tfidf_vectorizer = TfidfVectorizer(stop_words='english') 
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_words) 

        feature_names = tfidf_vectorizer.get_feature_names_out()
        word_indices = [feature_names.tolist().index(word) for word in all_words if word in feature_names.tolist()]
        filtered_tfidf_matrix = tfidf_matrix[:, word_indices]  

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(filtered_tfidf_matrix.toarray()) 
        df_pca = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "word": all_words,
            "cluster": all_labels})
        df_pca["cluster"] = df_pca["cluster"].astype(str)  
        fig = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="cluster", 
            text="word",  
            title=f"Clusters of words for the event : {event_type}",
            labels={"PC1": "PCA Component 1", "PC2": "PCA Component 2"},
            hover_data=["word"]
        )

        fig.update_traces(textposition='top center')  
        st.plotly_chart(fig) 



    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()

    #Extract tweets and their associated event types
    tweets_data = []
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            tweet_text = tweet_data.get("text", "")
            for u, v, edge_data in graph.edges(tweet_node, data=True):
                if edge_data.get("label") == "IS_ABOUT":
                    event_data = graph.nodes[v]
                    event_type = event_data.get("eventType", "Unknown")
                    tweets_data.append({"eventType": event_type, "tweetText": tweet_text})

    df_tweets = pd.DataFrame(tweets_data)
    
    # Select the event
    event_types = df_tweets['eventType'].unique()
    selected_event = st.selectbox("Select an event to analyze:", event_types)
    df_tweet_event = df_tweets[df_tweets["eventType"] == selected_event]

    if df_tweet_event.empty:
        st.write("No tweets available for the selected event.")
        st.stop()

    #1. Preprocessing 
    cleaned_tweets = [clean_tweet(tweet) for tweet in df_tweet_event["tweetText"]]

    #2. Compute the representation of TF-IDF
    tfidf_vectorizer, tfidf_matrix = get_tfidf_representations(cleaned_tweets)
    
    #3. Apply the clustering K-means
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)
    clusters, tfidf_vectorizer, tfidf_matrix = cluster_keywords_by_event(selected_event, cleaned_tweets, num_clusters)
    st.subheader(f"Word Clusters for {selected_event}")
    for cluster, words in clusters.items():
        st.write(f"Cluster {cluster + 1}: {', '.join(words)}")

    #4. Plot the word with the most frequency
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    plot_tfidf_keywords(tfidf_df, top_n=20)
    
    #5. Visualization t-SNE for the word
    vocab = tfidf_vectorizer.get_feature_names_out()
    tsne_tfidf = TSNE(n_components=2, random_state=42, perplexity=5, init='random', learning_rate=200)
    tfidf_2d = tsne_tfidf.fit_transform(tfidf_matrix.T.toarray())
    plot_single_points(vocab, {word: i for i, word in enumerate(vocab)}, tfidf_2d)

    #6. PCA for the clustering
    plot_word_clusters_PCA(selected_event, clusters)
    
    #7. Cosine similarity for words linked to the event
    words_of_interest = [selected_event]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_tweets)
    vocab = vectorizer.get_feature_names_out()
    word_indices = [vocab.tolist().index(word) for word in words_of_interest if word in vocab.tolist()]
    
    similarities = {}
    for word, index in zip(words_of_interest, word_indices):
        if index is not None:
            word_vector = tfidf_matrix[:, index].toarray().reshape(1, -1)
            cosine_sim = cosine_similarity(word_vector, tfidf_matrix.T).flatten()
            cosine_sim[index] = -1  # Exclure le mot lui-même
            most_similar_indices = cosine_sim.argsort()[-6:][::-1]
            similar_words = [(vocab[i], cosine_sim[i]) for i in most_similar_indices if i != index]
            similarities[word] = similar_words

    st.subheader(f"Top 5 Similar Words to '{selected_event}'")
    for word, similar_words in similarities.items():
        st.write(f"Top 5 similar words to '{word}':")
        for similar_word, score in similar_words:
            st.write(f"- {similar_word} (Similarity: {score:.4f})")



# =============================================================================
# PAGE "Word Embeddings"
# =============================================================================
elif page == "Word Embeddings":
    st.title("Word Embeddings")
    graph_path = os.path.join("database", "Everything","database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()
    tweets = [data['text'] for _, data in graph.nodes(data=True) if 'text' in data]
    if not tweets:
        st.error("No tweets found in the graph.")
    else:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        def clean_tweet(tweet: str) -> str:
            tweet = re.sub(r"http\S+|www\S+", '', tweet)
            tweet = re.sub(r'@\w+|#\w+', '', tweet)
            tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
            tweet = tweet.lower()
            tokens = word_tokenize(tweet)
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            return " ".join(tokens)
        cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
        tokenized_tweets = [tweet.split() for tweet in cleaned_tweets]
        @st.cache_data(show_spinner=True)
        def train_word2vec(tokenized_corpus: list[list[str]]) -> Word2Vec:
            return Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)
        word2vec_model = train_word2vec(tokenized_tweets)
        default_words = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']
        words = st.multiselect("Select at least 3 words to visualize", options=default_words, default=default_words)
        if len(words) < 3:
            st.error("Please select at least 3 words.")
        else:
            valid_words = [word for word in words if word in word2vec_model.wv]
            if not valid_words:
                st.error("None of the specified words were found in the model vocabulary.")
            else:
                word_vectors = [word2vec_model.wv[word] for word in valid_words]
                pca = PCA(n_components=2)
                word_vectors_pca = pca.fit_transform(word_vectors)
                df_vis = pd.DataFrame(word_vectors_pca, columns=["PC1", "PC2"])
                df_vis["word"] = valid_words
                num_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(word_vectors)
                df_vis["cluster"] = clusters.astype(str)
                fig = px.scatter(
                    df_vis,
                    x="PC1",
                    y="PC2",
                    color="cluster",
                    text="word",
                    title="Word Embeddings Visualization (PCA)",
                    hover_data=["word"]
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig)
                st.subheader("Word Similarity")
                st.write("Select words to compare their semantic similarity.")
                selected_word = st.selectbox("Choose a word to find similar words", valid_words)
                similar_words = word2vec_model.wv.most_similar(selected_word, topn=5)
                st.write(f"Top 5 words similar to **{selected_word}**:")
                for word, score in similar_words:
                    st.write(f"- **{word}** (Similarity: {score:.4f})")

# =============================================================================
# PAGE "Tweet Embeddings" 
# =============================================================================
elif page == "Tweet Embeddings":
    st.title("Tweet Embeddings")
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()
    tweet_list = []
    tweet_event_types = []
    for _, data in graph.nodes(data=True):
        if data.get("labels") == ":Tweet" and 'text' in data:
            tweet_list.append(data['text'])
            tweet_event_types.append(data.get("eventType", "Unknown"))
    # In the Tweet Embeddings page:
    if not tweet_list:
        st.error("No tweet embeddings found.")
    else:
        # Build the TF-IDF matrix (sparse)
        vectorizer, tfidf_matrix = build_tfidf(tweet_list)
        
        # Use TruncatedSVD to reduce dimensions before TSNE
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=50, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        
        # Now apply TSNE on the reduced dense array
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=15, n_components=2, init='pca', n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(tfidf_reduced)
        
        df_tsne = pd.DataFrame(tsne_results, columns=["x", "y"])
        df_tsne["eventType"] = tweet_event_types
        selected_event = st.selectbox("Select Event Type", df_tsne["eventType"].unique())
        filtered_df = df_tsne[df_tsne["eventType"] == selected_event]
        fig = px.scatter(filtered_df, x="x", y="y", title=f"Tweet Embeddings for {selected_event}")
        st.plotly_chart(fig)

# =============================================================================
# PAGE "Sentiment Analysis"
# =============================================================================
elif page == "Sentiment Analysis":
    st.title("Tweet Sentiment Analysis")
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    if not os.path.exists(graph_path):
        st.error(f"Graph file not found at {graph_path}. Please check your file path.")
        st.stop()
    
    graph = nx.read_graphml(graph_path)
    
    # -----------------------------
    # Re-create the event type mapping as in the Home page
    # -----------------------------
    topic_to_event_type = {}
    for event_node, event_data in graph.nodes(data=True):
        if event_data.get("labels") == ":Event":
            event_id = event_data.get("trecisid")
            event_type = event_data.get("eventType")
            if event_id and event_type:
                topic_to_event_type[event_id] = event_type

    # Update tweets with event type from topic mapping
    tweet_list = []
    for node, data in graph.nodes(data=True):
        if data.get("labels") == ":Tweet" and 'text' in data:
            tweet_id = data.get("id")
            text = data.get("text")
            tweet_topic = data.get("topic")
            if tweet_topic and tweet_topic in topic_to_event_type:
                data['eventType'] = topic_to_event_type[tweet_topic]
            else:
                data['eventType'] = "Unknown"
            tweet_list.append((tweet_id, text, data.get("eventType")))
    
    if not tweet_list:
        st.error("No tweets found in the graph.")
        st.stop()
        
    df_tweets = pd.DataFrame(tweet_list, columns=["Tweet ID", "Text", "Event Type"])
    
    # -----------------------------
    # Event Type Filter 
    # -----------------------------
    st.subheader("Filter Tweets by Event Type")
    event_types_filter = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']
    selected_event_types = st.multiselect("Select Event Types", options=event_types_filter, default=event_types_filter)
    if selected_event_types:
        df_tweets = df_tweets[df_tweets["Event Type"].isin(selected_event_types)]
    else:
        st.warning("Please select at least one event type")
    
    # -----------------------------
    # Sentiment Filtering
    # -----------------------------
    selected_sentiment = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
    sia = SentimentIntensityAnalyzer()
    df_tweets["Compound Score"] = df_tweets["Text"].apply(lambda text: sia.polarity_scores(text)["compound"])
    
    def classify_sentiment(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    df_tweets["Polarity"] = df_tweets["Compound Score"].apply(classify_sentiment)
    if selected_sentiment != "All":
        df_tweets = df_tweets[df_tweets["Polarity"] == selected_sentiment]
    
    if df_tweets.empty:
        st.warning("No tweets match the selected filters. Try selecting a different sentiment or event type.")
        st.stop()
        
    st.write("### Sample of Sentiment-Classified Tweets")
    st.dataframe(df_tweets.drop(columns=["Event Type"]).head(10))
    
    sentiment_counts = df_tweets["Polarity"].value_counts().reset_index()
    sentiment_counts.columns = ["Polarity", "Count"]
    if not sentiment_counts.empty:
        fig = px.bar(sentiment_counts, x="Polarity", y="Count",
                     title="Tweet Sentiment Distribution",
                     color="Polarity", template="plotly_white")
        st.plotly_chart(fig)
    else:
        st.warning("No sentiment data available for the selected tweets.")

# =============================================================================
# PAGE "Search System"
# =============================================================================
elif page == "Search System":
    st.title("Tweet Search System")
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()
    tweet_list = []
    for _, data in graph.nodes(data=True):
        if 'text' in data:
            tweet_list.append({"id": data.get("id"), "text": data.get("text")})
    df_tweets = pd.DataFrame(tweet_list)
    if df_tweets.empty:
        st.error("No tweets found in the graph.")
    else:
        st.write(f"Loaded {df_tweets.shape[0]} tweets.")
        nltk_stopwords = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        def preprocess_text(text: str) -> str:
            text = re.sub(r"http\S+|www\S+", '', text)
            text = re.sub(r'@\w+|#\w+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in nltk_stopwords]
            return " ".join(tokens)
        start_time = time.time()
        df_tweets["processed_text"] = df_tweets["text"].apply(preprocess_text)
        @st.cache_data(show_spinner=True)
        def build_tfidf(docs: list[str]):
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(docs)
            return vectorizer, tfidf
        vectorizer, tfidf_matrix = build_tfidf(df_tweets["processed_text"].tolist())
        query = st.text_input("Enter your search query", "earthquake rescue help")
        top_k = st.slider("Number of tweets to retrieve:", 1, 20, 5)
        if query:
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            st.subheader(f"Top-{top_k} Relevant Tweets")
            results = []
            for idx in top_indices:
                results.append({
                    "Tweet ID": df_tweets.iloc[idx]["id"],
                    "Text": df_tweets.iloc[idx]["text"],
                    "Relevance Score": f"{similarities[idx]:.4f}"
                })
            st.table(pd.DataFrame(results))
        st.subheader("Toy Dataset of Test Queries")
        toy_queries = [
            "earthquake damage relief",
            "flood emergency shelter",
            "wildfire smoke evacuation",
            "shooting police suspect",
            "typhoon wind power outage"
        ]
        st.write(toy_queries)