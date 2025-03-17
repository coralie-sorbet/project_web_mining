import os
import re
import time
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import streamlit as st

# --- Scikit-learn ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- NLTK ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# --- PyTorch & Transformers ---
import torch  
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import gensim.downloader as api

# =============================================================================
# UTILITY FUNCTIONS & CACHING
# =============================================================================

@st.cache_data(show_spinner=True)
def load_graph(path: str) -> nx.Graph:
    """Load the NetworkX graph from file using caching."""
    if not os.path.exists(path):
        st.error(f"Graph file not found at {path}. Please check your file path.")
        return None
    try:
        return nx.read_graphml(path)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None

@st.cache_data(show_spinner=True)
def compute_event_statistics(graph: nx.Graph) -> pd.DataFrame:
    """Compute aggregated tweet statistics per event type."""
    event_types = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']
    event_stats = []
    topic_to_event_type = {}
    tweet_dates_by_event = {etype: [] for etype in event_types}
    user_ids_by_event = {etype: set() for etype in event_types}

    # First pass: Build mapping for events and collect tweet dates and user IDs.
    for node, data in graph.nodes(data=True):
        label = data.get("labels", "")
        if label == ":Event":
            trecisid = data.get("trecisid", "")
            event_type = data.get("eventType")
            if trecisid and event_type:
                topic_to_event_type[trecisid] = event_type
        elif label == ":Tweet":
            tweet_topic = data.get("topic")
            if tweet_topic and tweet_topic in topic_to_event_type:
                event_type = topic_to_event_type[tweet_topic]
                data['eventType'] = event_type  # Add eventType to tweet node
                if 'created_at' in data:
                    tweet_dates_by_event[event_type].append(data['created_at'])
                # Here, we assume tweet "id" represents the user. Adjust if necessary.
                if 'id' in data:
                    user_ids_by_event[event_type].add(data['id'])

    # Second pass: Compute statistics for each event type.
    for etype in event_types:
        dates = pd.to_datetime(tweet_dates_by_event[etype], errors='coerce').dropna()
        if dates.empty:
            continue
        tweet_counts = pd.Series(dates).groupby(dates.dt.date).size()
        num_tweets = int(tweet_counts.sum())
        first_tweet = dates.min()
        last_tweet = dates.max()
        avg_freq = (last_tweet - first_tweet).days / num_tweets if num_tweets > 0 else 0
        event_stats.append({
            'Event_Type': etype,
            'Number_of_Tweets_perEvent': num_tweets,
            'Number_of_Users_perEvent': len(user_ids_by_event[etype]),
            'First_Tweet_Date': first_tweet,
            'Last_Tweet_Date': last_tweet,
            'Avg_Tweet_Frequency': avg_freq
        })
    return pd.DataFrame(event_stats)

@st.cache_data(show_spinner=True)
def build_tfidf(docs: list[str]) -> tuple[TfidfVectorizer, any]:
    """Build and cache the TF-IDF vectorizer and matrix."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)
    return vectorizer, tfidf

@st.cache_data(show_spinner=True)
def train_word2vec(tokenized_corpus: list[list[str]]) -> Word2Vec:
    """Train and cache a Word2Vec model."""
    return Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)

# Preprocessing function used for tweets
def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# =============================================================================
# STREAMLIT NAVIGATION
# =============================================================================
page = st.sidebar.radio("Select a page", (
    "Home",
    "TF-IDF",
    "Word Embeddings",
    "Tweet Embeddings",
    "Search System",
    "Sentiment Analysis"
))

# =============================================================================
# PAGE: HOME (Dashboard & Event Statistics)
# =============================================================================
if page == "Home":
    st.title("Welcome to the Event Dashboard")
    st.write("Here you can select events and view corresponding data.")
    
    # Load graph once using the cached function
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()

    # Build time series data for tweets by event type
    data = []
    event_types = ['typhoon', 'shooting', 'wildfire', 'bombing', 'earthquake', 'flood']
    for etype in event_types:
        tweet_dates = []
        for node, data_dict in graph.nodes(data=True):
            if data_dict.get("labels") == ":Tweet" and data_dict.get("eventType") == etype:
                if 'created_at' in data_dict:
                    tweet_dates.append(data_dict['created_at'])
        tweet_dates = pd.to_datetime(tweet_dates, errors='coerce').dropna()
        tweet_counts = pd.Series(tweet_dates).groupby(tweet_dates.dt.date).size()
        for date, count in tweet_counts.items():
            data.append({'date': date, 'Event_Type': etype, 'num_tweets_perday': count})
    df_tweets_time_series = pd.DataFrame(data)

    # Compute additional event statistics using the cached function
    event_stats_df = compute_event_statistics(graph)

    # Merge the time series data with event statistics
    df_complete = pd.merge(df_tweets_time_series, event_stats_df, on='Event_Type', how='left')

    # Create an aggregated resume table by event type
    df_resume = df_complete.groupby('Event_Type').agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max'),
        Number_of_Users_perEvent=('Number_of_Users_perEvent', 'max'),
        First_Tweet_Date=('date', 'min'),
        Last_Tweet_Date=('date', 'max'),
        Avg_Tweet_Frequency=('Avg_Tweet_Frequency', 'max')
    ).reset_index()

    st.sidebar.write('Select Filter')
    # Use session state for selected events to prevent re-runs
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = ['typhoon']  # Pre-select an event

    choices = list(df_complete['Event_Type'].unique())
    col_count = len(choices)
    columns = st.columns(col_count)
    for idx, event in enumerate(choices):
        is_selected = event in st.session_state.selected_events
        # Color button style based on selection (for illustration only)
        with columns[idx]:
            if st.button(event, key=event, help=f'Select {event}'):
                if is_selected:
                    st.session_state.selected_events.remove(event)
                else:
                    st.session_state.selected_events.append(event)

    if st.session_state.selected_events:
        st.write(f"### Data for selected events: {', '.join(st.session_state.selected_events)}")
        df_resume_filtered = df_resume[df_resume['Event_Type'].isin(st.session_state.selected_events)]
        st.dataframe(df_resume_filtered)
    else:
        st.write("Please select at least one event type")

    # Filter complete dataset based on selection and aggregate per day
    df_flt = df_complete[df_complete['Event_Type'].isin(st.session_state.selected_events)]
    df_flt['date'] = pd.to_datetime(df_flt['date'])
    df_flt = df_flt.groupby(['Event_Type', 'date']).agg(
        Number_of_Tweets_per_day=('num_tweets_perday', 'max')
    ).reset_index()

    # Create and display the temporal plot
    fig = px.line(
        df_flt, 
        x='date', 
        y='Number_of_Tweets_per_day', 
        color='Event_Type', 
        title="Evolution of Tweets per Day by Event Type",
        labels={'num_tweets_perday': 'Tweets per Day', 'date': 'Date'},
        line_shape='linear',
        markers=True,
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        legend_title="Event Type",
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14)
    )
    st.plotly_chart(fig)
    st.dataframe(df_flt if not df_flt.empty else "Empty Dataframe")

# =============================================================================
# PAGE: TF-IDF
# =============================================================================
elif page == "TF-IDF":
    st.title("TF-IDF Page")
    st.write("Welcome to the TF-IDF demonstration.")

    # Load the graph
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()

    # Extract tweets from graph
    tweet_list = []
    for _, data in graph.nodes(data=True):
        if 'text' in data:
            tweet_list.append({"id": data.get("id"), "text": data.get("text")})
    df_tweets = pd.DataFrame(tweet_list)
    if df_tweets.empty:
        st.error("No tweets found in the graph.")
    else:
        st.write(f"Loaded {df_tweets.shape[0]} tweets.")
        # Preprocess text and cache results if possible
        df_tweets["processed_text"] = df_tweets["text"].apply(preprocess_text)
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

# =============================================================================
# PAGE: Word Embeddings
# =============================================================================
elif page == "Word Embeddings":
    st.title("Word Embeddings")
    # Load graph and extract tweets
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()
    tweets = [data.get('text', '') for _, data in graph.nodes(data=True) if 'text' in data]
    if not tweets:
        st.error("No tweets found in the graph.")
    else:
        # Preprocess tweets and tokenize
        cleaned_tweets = [preprocess_text(tweet) for tweet in tweets]
        tokenized_tweets = [tweet.split() for tweet in cleaned_tweets]

        # Train Word2Vec model (cached)
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
                st.write("Select a word to view similar words.")
                selected_word = st.selectbox("Choose a word", valid_words)
                similar_words = word2vec_model.wv.most_similar(selected_word, topn=5)
                st.write(f"Top 5 words similar to **{selected_word}**:")
                for word, score in similar_words:
                    st.write(f"- **{word}** (Similarity: {score:.4f})")

# =============================================================================
# PAGE: Tweet Embeddings
# =============================================================================
elif page == "Tweet Embeddings":
    st.title("Tweet Embeddings")
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    graph = load_graph(graph_path)
    if graph is None:
        st.stop()

    # Example: Use a cached TF-IDF embedding approach (or similar) to embed tweets.
    # The following code is a placeholder and can be extended similarly to Word Embeddings.
    tweet_list = []
    tweet_event_types = []
    for _, data in graph.nodes(data=True):
        if data.get("labels") == ":Tweet" and 'text' in data:
            tweet_list.append(data['text'])
            tweet_event_types.append(data.get("eventType", "Unknown"))
    if not tweet_list:
        st.error("No tweet embeddings found.")
    else:
        # For demonstration, we reuse the TF-IDF approach to create embeddings.
        vectorizer, tfidf_matrix = build_tfidf(tweet_list)
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=15, n_components=2, init='pca', n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
        df_tsne = pd.DataFrame(tsne_results, columns=["x", "y"])
        df_tsne["eventType"] = tweet_event_types
        selected_event = st.selectbox("Select Event Type", df_tsne["eventType"].unique())
        filtered_df = df_tsne[df_tsne["eventType"] == selected_event]
        fig = px.scatter(filtered_df, x="x", y="y", title=f"Tweet Embeddings for {selected_event}")
        st.plotly_chart(fig)

# =============================================================================
# PAGE: Sentiment Analysis
# =============================================================================
elif page == "Sentiment Analysis":
    st.title("Tweet Sentiment Analysis")
    graph_path = os.path.join("database", "Everything", "database_formated_for_NetworkX.graphml")
    if not os.path.exists(graph_path):
        st.error(f"Graph file not found at {graph_path}. Please check your file path.")
        st.stop()
    graph = load_graph(graph_path)

    tweet_list = []
    for _, data in graph.nodes(data=True):
        if 'text' in data:
            tweet_id = data.get("id")
            text = data.get("text")
            event_type = data.get("eventType", "Unknown")
            tweet_list.append((tweet_id, text, event_type))
    if not tweet_list:
        st.error("No tweets found in the graph.")
        st.stop()

    df_tweets = pd.DataFrame(tweet_list, columns=["Tweet ID", "Text", "Event Type"])

    selected_sentiment = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])

    # Use VADER for sentiment analysis
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
        st.warning("No tweets match the selected sentiment. Try selecting a different sentiment.")
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

