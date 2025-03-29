import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import hvplot.pandas
import streamlit as st
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import gensim.downloader as api

import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# --- NLTK Setup ---
nltk_data_path = os.environ.get("NLTK_DATA_PATH", '/tmp/nltk_data')
os.environ["NLTK_DATA"] = nltk_data_path

# Ensure the nltk_data directory exists
os.makedirs(nltk_data_path, exist_ok=True)

# Append the nltk_data_path to the NLTK data search paths if not already present
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Function to download missing NLTK resources
def download_nltk_resources(resources):
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_path, quiet=True)

# List of NLTK resources needed
resources = ["stopwords", "punkt", "wordnet", "vader_lexicon", "sentiwordnet"]
download_nltk_resources(resources)

# Ensure necessary tokenizer resources are downloaded
tokenizer_resources = ["punkt", "punkt_tab"]
for resource in tokenizer_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)

# --- NLTK Tokenization and Sentiment Analysis ---
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet as wn, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# --- TF-IDF Vectorization ---
@st.cache_data(show_spinner=True)
def build_tfidf(docs: list[str]):
    """Build and return TF-IDF matrix for the given documents."""
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

# =============================================================================
# Load the graph data
# =============================================================================
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
# PAGE "Home": Dashboard and Event Statistics
# =============================================================================
if page == "Home":
    st.title("Welcome to the Event Dashboard")
    st.write("Here you can select events and view the corresponding data.")

    # Path to the GraphML file
    path_data = "database/Everything/database_formated_for_NetworkX.graphml"

    # Load the graph using a cached function
    graph = load_graph(path_data)
    if graph is None:
        st.stop()

    # Extract unique labels from the graph nodes
    unique_labels = set(data.get("labels") for _, data in graph.nodes(data=True))

    # Create a dictionary to store the correspondence between "topic" and the "inter" attribute of an Event node
    topic_to_inter = {}

    # Create a set for unique topics
    unique_topics = set()

    # Iterate through all nodes of type "Tweet" to extract unique topics
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            topic = tweet_data.get("topic")
            if topic:
                unique_topics.add(topic)

    # For each unique topic, associate the event and its "inter" attribute
    for topic in unique_topics:
        for event_node, event_data in graph.nodes(data=True):
            if event_data.get("labels") == ":Event" and topic in event_data.get("trecisid", ""):
                event_id = event_data.get("id")
                topic_to_inter[topic] = event_id

    # Create a dictionary to store the correspondence between Tweet and User
    tweet_to_user = {}
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("label") == "POSTED":
            user_id = graph.nodes[u].get("id")
            tweet_id = graph.nodes[v].get("id")
            tweet_to_user[tweet_id] = user_id

    # Dictionary to associate tweet topics with their EventType
    topic_to_event_type = {}
    for event_node, event_data in graph.nodes(data=True):
        if event_data.get("labels") == ":Event":
            event_id = event_data.get("trecisid")
            event_type = event_data.get("eventType")
            if event_id and event_type:
                topic_to_event_type[event_id] = event_type

    # Associate each tweet with its EventType based on its topic
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            tweet_topic = tweet_data.get("topic")
            if tweet_topic and tweet_topic in topic_to_event_type:
                tweet_data['eventType'] = topic_to_event_type[tweet_topic]

    # Dictionary to count tweets by event type
    event_type_count = {}
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet" and 'eventType' in tweet_data:
            event_type = tweet_data['eventType']
            if event_type:
                event_type_count[event_type] = event_type_count.get(event_type, 0) + 1

    # List of event types
    event_types = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']

    # Store data for the time series (number of tweets per day per event type)
    time_series_data = []
    for event_type in event_types:
        tweet_dates = []
        for tweet_node, tweet_data in graph.nodes(data=True):
            if tweet_data.get("labels") == ":Tweet" and tweet_data.get("eventType") == event_type:
                if 'created_at' in tweet_data:
                    tweet_dates.append(tweet_data['created_at'])
        # Convert dates to datetime and remove NaT values
        tweet_dates = pd.to_datetime(tweet_dates, errors='coerce').dropna()
        # Create a series of dates (conversion to date)
        tweet_counts = pd.Series([x.date() for x in tweet_dates]).value_counts().sort_index()
        for date, count in tweet_counts.items():
            time_series_data.append({'date': date, 'Event_Type': event_type, 'num_tweets_perday': count})

    df_tweets_time_series = pd.DataFrame(time_series_data)
    # Optionally, display the first rows: st.dataframe(df_tweets_time_series.head())

    # Define event colors (for potential use)
    event_colors = {
        'typhoon': 'royalblue',
        'shooting': 'darkorange',
        'wildfire': 'green',
        'bombing': 'red',
        'earthquake': 'purple',
        'flood': 'cyan'
    }

    # List to store statistics for each event type
    event_stats = []

    # List to store data for the "IS_ABOUT" relationships
    is_about_data = []
    for u, v, edge_data in graph.edges(data=True):
        if edge_data.get("label") == "IS_ABOUT":
            is_about_data.append({"user": u, "event": v, "label": "IS_ABOUT"})
    df_is_about = pd.DataFrame(is_about_data)

    # List to store event data
    event_data_list = []
    for node, data in graph.nodes(data=True):
        if data.get("labels") == ":Event":
            event_data_list.append({"event": node, "event_id": data.get("id"), "eventType": data.get("eventType")})
    df_events = pd.DataFrame(event_data_list)

    # Merge DataFrames on the "event" column
    df_combined = pd.merge(df_is_about, df_events, on='event', how='left')

    # Group by "eventType" and count the number of unique users
    user_counts_by_event = df_combined.groupby('eventType')['user'].nunique().reset_index()
    subevent_counts = df_combined.groupby(['eventType'])['event_id'].nunique().reset_index(name='unique_event_id_count')

    user_counts_by_event = user_counts_by_event.rename(columns={"user": "Number of Users"})
    subevent_counts = subevent_counts.rename(columns={"event_id": "Number of Unique Events in the Category"})

    # Calculate statistics for each event type
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
        num_subevent = subevent_counts[subevent_counts['eventType'] == event_type]['unique_event_id_count'].values[0]
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

    st.title("Temporal Distribution of Tweets for Each Event Type")
    st.sidebar.write('Select Filter')

    # Create event filter checkboxes with "wildfire" selected by default
    choices = list(df_complete['Event_Type'].unique())
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = ['wildfire']

    col_count = len(choices)
    columns = st.columns(col_count)
    for idx, event in enumerate(choices):
        with columns[idx]:
            checked = st.checkbox(event, value=(event in st.session_state.selected_events), key=f"checkbox_{event}")
            if checked and event not in st.session_state.selected_events:
                st.session_state.selected_events.append(event)
            elif not checked and event in st.session_state.selected_events:
                st.session_state.selected_events.remove(event)

    if st.session_state.selected_events:
        st.write(f"### Data for Selected Events: {', '.join(st.session_state.selected_events)}")
        df_resume_filtered = df_resume[df_resume['Event_Type'].isin(st.session_state.selected_events)]
        st.dataframe(df_resume_filtered)
    else:
        st.write("Please select at least one event type")

    df_complete['date'] = pd.to_datetime(df_complete['date'])
    df_flt = df_complete[df_complete['Event_Type'].isin(st.session_state.selected_events)]
    df_flt = df_flt.groupby(['Event_Type', 'date']).agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max')
    ).reset_index()

    # Line chart: Evolution of the number of tweets posted each day by event type
    fig = px.line(df_flt, 
                  x='date', 
                  y='Number_of_Tweets_per_day', 
                  color='Event_Type', 
                  title="Evolution of the Number of Tweets Posted Each Day by Event Type",
                  labels={'Number_of_Tweets_per_day': 'Number of Tweets per Day', 'date': 'Date'},
                  line_shape='linear',
                  markers=True,
                  template="plotly_white")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Tweets Posted",
        legend_title="Event Type",
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
    )
    st.plotly_chart(fig)
                                
    if df_flt.shape[0] > 0:
        st.dataframe(df_flt)
    else:
        st.write("Empty DataFrame")

    # ----------------------------------------------------------------------------
    # Create a DataFrame of tweet nodes for further analysis (if not already available)
    # ----------------------------------------------------------------------------
    tweets_records = []
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            record = tweet_data.copy()
            record["node"] = tweet_node
            tweets_records.append(record)
    df_tweet_event = pd.DataFrame(tweets_records)

    # ----------------------------------------------------------------------------
    # Temporal Distribution of Words Across the Event Timeline
    # ----------------------------------------------------------------------------
    st.title("Temporal Distribution of Words Across the Event Timeline")
    # Use the cleaned tweet texts for word distribution analysis.
    # Use the appropriate column name for tweet text if necessary.
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    import re
    import nltk
    nltk.download("punkt", download_dir=nltk_data_path)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union({"http", "https", "rt", "news", "amp", "nhttps"})
    def clean_tweet(tweet: str) -> str:
        tweet = re.sub(r"http\S+|www\S+", '', tweet)
        tweet = re.sub(r'@\w+|#\w+', '', tweet)
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = tweet.lower()
        tokens = word_tokenize(tweet)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    if "cleanedText" not in df_tweet_event.columns:
        content_column = "tweetText" if "tweetText" in df_tweet_event.columns else "text"
        df_tweet_event["cleanedText"] = df_tweet_event[content_column].apply(lambda t: clean_tweet(t))

    # Convert 'created_at' to datetime if not already done.
    df_tweet_event['created_at'] = pd.to_datetime(df_tweet_event['created_at'], errors='coerce')

    # Filter tweets for selected events
    df_words = df_tweet_event[df_tweet_event['eventType'].isin(st.session_state.selected_events)]
    st.write("Filtered tweets for word analysis:", df_words.shape)

    # Create a list of records for words with their corresponding date
    word_records = []
    for _, row in df_words.iterrows():
        if pd.notnull(row['created_at']) and row['cleanedText']:
            words = row['cleanedText'].split()
            for word in words:
                word_records.append({'date': row['created_at'].date(), 'word': word})
    df_words_time = pd.DataFrame(word_records)
    st.write("Words time DataFrame shape:", df_words_time.shape)

    # Compute overall top 10 words for the selected events
    if not df_words_time.empty:
        top_words = df_words_time['word'].value_counts().head(10).index.tolist()
        st.write("Top words:", top_words)
    else:
        st.write("No word records found. Check your tweet data and cleaning function.")

    # Filter the DataFrame for only top words
    df_top_words = df_words_time[df_words_time['word'].isin(top_words)]
    # Group by date and word and count occurrences
    df_top_words_time = df_top_words.groupby(['date', 'word']).size().reset_index(name='count')
    st.write("Grouped word time DataFrame shape:", df_top_words_time.shape)
    st.dataframe(df_top_words_time.head())

    # Plot the temporal distribution of top words if data is available
    if not df_top_words_time.empty:
        fig_words = px.line(df_top_words_time, 
                            x='date', 
                            y='count', 
                            color='word', 
                            title="Temporal Distribution of Top Words in Tweets",
                            labels={'count': 'Word Count', 'date': 'Date', 'word': 'Word'},
                            markers=True,
                            template="plotly_white")
        fig_words.update_layout(
            xaxis_title="Date",
            yaxis_title="Word Count",
            legend_title="Word",
            hovermode="x unified",
            font=dict(family="Arial, sans-serif", size=14),
        )
        st.plotly_chart(fig_words)
    else:
        st.write("No data available to plot the temporal distribution of words.")

  
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
                    labels={'word': 'Words', 'tfidf_score': 'TF-IDF Score'},
                    color='tfidf_score', 
                    color_discrete_map={'high': '#1f77b4', 'medium': '#66b3ff', 'low': '#cce5ff'})  

        fig.update_layout(xaxis_title="Words", yaxis_title="TF-IDF Score")
        st.plotly_chart(fig) 
    
    # --- Visualise words with TSNE ---
    def plot_single_points(all_words, word_to_index, vis_2d, selected_event):
        df_vis = pd.DataFrame({
            "word": all_words,
            "x": [vis_2d[word_to_index[word], 0] for word in all_words],
            "y": [vis_2d[word_to_index[word], 1] for word in all_words],
            "color": ["Selected event word" if word == selected_event else "Other Word" for word in all_words]})

        fig = px.scatter(
            df_vis,
            x="x", 
            y="y", 
            text="word", 
            labels={"x": "t-SNE X", "y": "t-SNE Y", "color": "Label"}, 
            hover_data=["word"], 
            color="color",  
            color_discrete_map={"Selected event word": "red", "Other Word": "blue"} 
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

        return word_clusters
    
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
    selected_event = st.selectbox("Select an event :", event_types)
    df_tweet_event = df_tweets[df_tweets["eventType"] == selected_event]

    if df_tweet_event.empty:
        st.write("No tweets available for the selected event")
        st.stop()

    #1. Preprocessing 
    cleaned_tweets = [clean_tweet(tweet) for tweet in df_tweet_event["tweetText"]]

    #2. Plot the word with the most frequency
    tfidf_vectorizer, tfidf_matrix=get_tfidf_representations(cleaned_tweets)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    st.subheader(f"Top 20 TF-IDF Keywords")
    plot_tfidf_keywords(tfidf_df, top_n=20)

    #3. Cosine similarity for words linked to the event
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
    
    #4. Visualization t-SNE for the word
    vocab = tfidf_vectorizer.get_feature_names_out()
    tsne_tfidf = TSNE(n_components=2, random_state=42, perplexity=5, init='random', learning_rate=200)
    tfidf_2d = tsne_tfidf.fit_transform(tfidf_matrix.T.toarray())
    st.subheader(f"First 100 Word Vector Representations")
    plot_single_points(vocab, {word: i for i, word in enumerate(vocab)}, tfidf_2d,selected_event)

    #5. Apply the clustering K-means
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)
    clusters = cluster_keywords_by_event(selected_event, cleaned_tweets, num_clusters)
    st.subheader(f"Word Clusters for {selected_event}")
    for cluster, words in clusters.items():
        st.write(f"Cluster {cluster + 1}: {', '.join(words)}")

    #6. PCA for the clustering
    st.subheader(f"Word clusters after PCA for event '{selected_event}'")
    plot_word_clusters_PCA(selected_event, clusters)



# =============================================================================
# PAGE "Word Embeddings"
# =============================================================================
elif page == "Word Embeddings":
    st.title("Word Embeddings Analysis")
    st.write("Here we analyze the words in tweets linked to different types of events using Word Embeddings.")

    # --- Load Graph Data ---
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()

    # --- Extract tweets and event types ---
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
    event_types = df_tweets['eventType'].unique()
    selected_event = st.selectbox("Select an event:", event_types)
    df_tweet_event = df_tweets[df_tweets["eventType"] == selected_event]

    if df_tweet_event.empty:
        st.write("No tweets available for the selected event.")
        st.stop()

    # --- Text Preprocessing ---
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union({"http", "https", "rt", "news", "amp", "nhttps"})

    def clean_tweet(tweet: str) -> str:
        tweet = re.sub(r"http\S+|www\S+", '', tweet)
        tweet = re.sub(r'@\w+|#\w+', '', tweet)
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = tweet.lower()
        tokens = word_tokenize(tweet)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    cleaned_tweets = [clean_tweet(tweet) for tweet in df_tweet_event["tweetText"]]
    tokenized_tweets = [tweet.split() for tweet in cleaned_tweets]

    # --- Train Word2Vec Model ---
    @st.cache_data(show_spinner=True)
    def train_word2vec(tokenized_corpus: list[list[str]]) -> Word2Vec:
        return Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)
    
    word2vec_model = train_word2vec(tokenized_tweets)

    # --- Filter Vocabulary based on Frequency ---
    from collections import Counter
    word_freq = Counter(word for tweet in tokenized_tweets for word in tweet)
    # Keep only words present in the model meeting the frequency threshold
    filtered_vocab = [word for word in word2vec_model.wv.index_to_key if word_freq[word] >= 3]

    # --- Additional Filtering: Limit number of words ---
    # Sort by frequency descending and select the top 'max_words'
    filtered_vocab = sorted(filtered_vocab, key=lambda w: word_freq[w], reverse=True)[:100]

    if not filtered_vocab:
        st.warning("No words meet the frequency criteria. Please adjust the minimum frequency.")
    else:
        # --- Word Clustering on Filtered Vocabulary ---
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)
        filtered_vectors = np.array([word2vec_model.wv[word] for word in filtered_vocab])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(filtered_vectors)
        
        df_clusters = pd.DataFrame({"word": filtered_vocab, "cluster": clusters})
        st.subheader(f"Word Clusters for {selected_event}")
        st.write("The top 20 words in each cluster, ranked by their Word Embedding scores.")
        for cluster in range(num_clusters):
            words_in_cluster = df_clusters[df_clusters["cluster"] == cluster]["word"].tolist()
            # Display only the first 20 words per cluster for better readability
            st.write(f"Cluster {cluster + 1}: {', '.join(words_in_cluster[:20])}")
        

        # --- Visualization with PCA for Clusters ---
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(filtered_vectors)
        df_pca = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "word": filtered_vocab,
            "cluster": clusters.astype(str)
        })
        fig_pca = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="cluster",
            text="word",
            title=f"PCA of Word Embeddings Clusters for {selected_event}, displaying the top 100 words.",
            hover_data=["word"]
        )
        fig_pca.update_traces(textposition='top center')
        st.plotly_chart(fig_pca)
        
        # --- Word Similarity ---
        st.subheader("Word Similarity")
        similar_words = word2vec_model.wv.most_similar(selected_event, topn=5)
        st.write(f"Top 5 words similar to **{selected_event}**:")
        for word, score in similar_words:
            st.write(f"- **{word}** (Similarity: {score:.4f})")

# =============================================================================
# PAGE "Tweet Embeddings" 
# =============================================================================
elif page == "Tweet Embeddings":
    st.title("Tweet Embeddings Analysis")
    st.write("Here we analyze the words in tweets linked to different types of events using Tweet Embeddings.")

     # --- Load Graph Data ---
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()

    # --- Extract tweets and event types ---
    tweets_data = []
    for tweet_node, tweet_data in graph.nodes(data=True):
        if tweet_data.get("labels") == ":Tweet":
            tweet_text = tweet_data.get("text", "")
            for u, v, edge_data in graph.edges(tweet_node, data=True):
                if edge_data.get("label") == "IS_ABOUT":
                    event_data = graph.nodes[v]
                    event_type = event_data.get("eventType", "Unknown")
                    tweets_data.append({"eventType": event_type, "tweetText": tweet_text, "tweetID": tweet_node})


    df_tweets = pd.DataFrame(tweets_data)
    event_types = df_tweets['eventType'].unique()
    selected_event = st.selectbox("Select an event:", event_types)
    df_tweet_event = df_tweets[df_tweets["eventType"] == selected_event]

    if df_tweet_event.empty:
        st.write("No tweets available for the selected event.")
        st.stop()

    # --- Text Preprocessing ---
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union({"http", "https", "rt", "news", "amp", "nhttps"})

    def clean_tweet(tweet: str) -> str:
        tweet = re.sub(r"http\S+|www\S+", '', tweet)
        tweet = re.sub(r'@\w+|#\w+', '', tweet)
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = tweet.lower()
        tokens = word_tokenize(tweet)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    cleaned_tweets = [clean_tweet(tweet) for tweet in df_tweet_event["tweetText"]]
    tokenized_tweets = [tweet.split() for tweet in cleaned_tweets]

    # --- Train Word2Vec Model ---
    @st.cache_data(show_spinner=True)
    def train_word2vec(tokenized_corpus: list[list[str]]) -> Word2Vec:
        return Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)
   
    word2vec_model = train_word2vec(tokenized_tweets)

    # --- Compute Tweet Embeddings ---
    def get_tweet_embedding(tweet: list[str], model: Word2Vec) -> np.ndarray:
        word_embeddings = []
        for word in tweet:
            if word in model.wv:
                word_embeddings.append(model.wv[word])
        if word_embeddings:
        # We take the mean of each word embeddings
            return np.mean(word_embeddings, axis=0)
        else:
            # If no words in the tweet have embeddings, return a zero vector
            return np.zeros(model.vector_size)

    tweet_embeddings = np.array([get_tweet_embedding(tweet, word2vec_model) for tweet in tokenized_tweets])

    # --- Clustering the Tweet Embeddings ---
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tweet_embeddings)
   
    df_clusters = pd.DataFrame({"tweet": df_tweet_event["tweetText"], "cluster": clusters})
    st.subheader(f"Tweet Clusters for {selected_event}")
    st.write("The top tweets in each cluster based on their Tweet Embedding scores.")
    for cluster in range(num_clusters):
        tweets_in_cluster = df_clusters[df_clusters["cluster"] == cluster]["tweet"].tolist()
        st.write(f"Cluster {cluster + 1}:")
        for tweet in tweets_in_cluster[:5]:  # Display only the first 5 tweets per cluster
            st.write(f"- {tweet}")

    # --- Visualization with PCA for Clusters ---
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(tweet_embeddings)
    df_pca = pd.DataFrame({
        "PC1": pca_components[:, 0],
        "PC2": pca_components[:, 1],
        "tweet": df_tweet_event["tweetText"],
        "tweetID": df_tweet_event["tweetID"],  
        "cluster": clusters.astype(str)
    })

    # Visualization of restricted number of tweets on the graph :
    N = 5
    df_pca_filtered = df_pca.groupby("cluster").head(N)

    df_pca_filtered['cluster'] = df_pca_filtered['cluster'].astype(int) + 1
    df_pca_filtered['cluster'] = df_pca_filtered['cluster'].astype(str)

    fig_pca = px.scatter(
        df_pca_filtered,
        x="PC1",
        y="PC2",
        color="cluster",
        text="tweetID",
        title=f"PCA of Tweet Embeddings Clusters for {selected_event} (max {N} tweets per cluster for better visualization )",
        hover_data=["tweet"]
    )

    fig_pca.update_traces(textposition='top center')
    st.plotly_chart(fig_pca)

    # --- Tweet Similarity ---
    st.subheader("Tweet Similarity")
    tweet_embedding = get_tweet_embedding(tokenized_tweets[0], word2vec_model)  # Get embedding of the first tweet
    similar_tweets = []
    for i, tweet in enumerate(tokenized_tweets):
        embedding = get_tweet_embedding(tweet, word2vec_model)
        similarity = cosine_similarity([tweet_embedding], [embedding])[0][0]
        similar_tweets.append((df_tweet_event.iloc[i]["tweetText"], similarity))
   
    similar_tweets = sorted(similar_tweets, key=lambda x: x[1], reverse=True)[:5]
    st.write(f"Top 5 similar tweets to the first tweet in the selected event:")
    for tweet, score in similar_tweets:
        st.write(f"- **{tweet}** (Similarity: {score:.4f})")

# =============================================================================
# PAGE "Sentiment Analysis"
# =============================================================================

elif page == "Sentiment Analysis":
    st.title("Tweet Sentiment Analysis")

    # Loading the graph
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    if not os.path.exists(graph_path):
        st.error(f"Graph file not found at {graph_path}. Please check your file path.")
        st.stop()
    
    graph = nx.read_graphml(graph_path)

    # Extract tweets and their event types
    tweet_list = []
    for _, data in graph.nodes(data=True):
        if 'text' in data:
            tweet_id = data.get("id")
            text = data.get("text")
            event_type = data.get("eventType", "Unknown")  # Default to "Unknown" if not found
            tweet_list.append((tweet_id, text, event_type))

    if not tweet_list:
        st.error("No tweets found in the graph.")
        st.stop()
    
    df_tweets = pd.DataFrame(tweet_list, columns=["Tweet ID", "Text", "Event Type"])

    # --- Polarity filter ---
    selected_sentiment = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])

    # --- Sentiment Analysis using VADER ---
    sia = SentimentIntensityAnalyzer()
    
    df_tweets["Compound Score"] = df_tweets["Text"].apply(lambda text: sia.polarity_scores(text)["compound"])

    # Classify sentiment based on compound score
    # Define a function to classify sentiment
    def classify_sentiment(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df_tweets["Polarity"] = df_tweets["Compound Score"].apply(classify_sentiment)

    # Filter tweets based on selected sentiment
    if selected_sentiment != "All":
        df_tweets = df_tweets[df_tweets["Polarity"] == selected_sentiment]

    # Check if the filtered DataFrame is empty
    if df_tweets.empty:
        st.warning("No tweets match the selected sentiment. Try selecting a different sentiment.")
        st.stop()

    # --- Display the filtered DataFrame ---
    st.write("### Sample of Sentiment-Classified Tweets")
    st.dataframe(df_tweets.drop(columns=["Event Type"]).head(10))

    # --- Display the sentiment distribution ---
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