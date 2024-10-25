import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
import joblib
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")

dataset_file = "../dataset/ProcessedCommentsJan2017.csv"
data = pd.read_csv(dataset_file)

analyzer = SentimentIntensityAnalyzer()

def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return ' '.join(nouns)

def sentiment_analysis(comment):
    sentiment = analyzer.polarity_scores(comment)
    return np.array([sentiment['neg'], sentiment['neu'], sentiment['pos']])

def weighted_tfidf_with_sentiment(tfidf_vector, sentiment_vector):
    pos_weight = sentiment_vector[2]
    neg_weight = sentiment_vector[0]
    return tfidf_vector * (pos_weight + 1.0)

comments_text = data['commentBody'].apply(extract_nouns)
sentiment_scores = np.array([sentiment_analysis(comment) for comment in data['commentBody']])

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

tfidf_matrix = vectorizer.fit_transform(comments_text)

weighted_tfidf_matrix = np.array([weighted_tfidf_with_sentiment(tfidf.toarray()[0], sentiment_scores[i])
                                  for i, tfidf in enumerate(tfidf_matrix)])

weighted_tfidf_matrix = normalize(weighted_tfidf_matrix)

X = weighted_tfidf_matrix

y = np.array([data['neg'], data['neu'], data['pos']]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputClassifier(MultinomialNB())

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy for multi-label prediction: {accuracy}")

joblib.dump(model, '../models/sentiment_nb_model.pkl')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")
