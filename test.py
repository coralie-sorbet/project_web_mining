
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Lecture du fichier graphml
pathData = "database_formated_for_NetworkX.graphml"
graph = nx.read_graphml(pathData)

# Fonction de nettoyage des tweets
def clean_tweet(tweet: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union({"http", "https", "rt", "news", "amp", "nhttps"})
    
    tweet = re.sub(r"http\S+|www\S+", '', tweet)  # Supprimer les URLs
    tweet = re.sub(r'@\w+|#\w+', '', tweet)  # Supprimer les mentions et hashtags
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  # Supprimer les caractères non alphabétiques
    tweet = tweet.lower()  # Convertir en minuscule
    tokens = word_tokenize(tweet)  # Tokenisation
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization et suppression des stopwords
    return " ".join(tokens)

# Obtenir les représentations TF-IDF
def get_tfidf_representations(texts):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix

# Plotting des points 2D avec t-SNE
def plot_single_points(all_words, word_to_index, vis_2d):
    plt.figure(figsize=(12, 8))
    for word in all_words:
        i = word_to_index[word]
        x, y = vis_2d[i, 0], vis_2d[i, 1]
        plt.scatter(x, y, color='blue', s=50)
        plt.text(x + 0.1, y + 0.1, word, fontsize=10)
    plt.title("Word Vector Representations", fontsize=14)
    plt.xlabel("t-SNE X", fontsize=12)
    plt.ylabel("t-SNE Y", fontsize=12)
    plt.grid(True)
    st.pyplot()

# Affichage des mots clés les plus importants
def plot_tfidf_keywords(tfidf_df, top_n=10):
    top_keywords = tfidf_df.mean(axis=0).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    top_keywords.plot(kind='bar', color='royalblue')
    plt.title('Top TF-IDF Keywords')
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Score')
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

# Lecture des tweets et des événements
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

# Interface Streamlit
st.title("TF-IDF Analysis of Tweets")
st.write("Sélectionnez un type d'événement pour voir l'analyse TF-IDF des tweets associés.")

# Sélection de l'événement
event_type = st.selectbox('Sélectionner un type d\'événement', event_types)

# Filtrer les tweets pour l'événement sélectionné
df_tweet_event = df_tweets[df_tweets["eventType"] == event_type]
tweets = df_tweet_event["tweetText"]
cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]

# Obtenir les représentations TF-IDF
tfidf_vectorizer, tfidf_matrix = get_tfidf_representations(cleaned_tweets)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Vocabulaire et t-SNE
vocab = tfidf_vectorizer.get_feature_names_out()
tsne_tfidf = TSNE(n_components=2, random_state=42, perplexity=5, init='random', learning_rate=200)
tfidf_2d = tsne_tfidf.fit_transform(tfidf_matrix.T.toarray())
word_to_index = {word: i for i, word in enumerate(vocab)}

# Afficher les résultats TF-IDF
plot_single_points(vocab, word_to_index, tfidf_2d)
plot_tfidf_keywords(tfidf_df, top_n=20)
