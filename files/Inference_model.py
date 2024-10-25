import joblib
import numpy as np

# Load the model and vectorizer
model_file = "../models/sentiment_nb_model.pkl"
vectorizer_file = "../models/tfidf_vectorizer.pkl"
model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

print(f"Model loaded {model_file}")
print(f"TF-IDF vectorizer loaded {vectorizer_file}")


def reverse_inference(model, vectorizer, keyword, sentiment_label):
    input_vector = vectorizer.transform([keyword])
    prediction_proba = model.predict_proba(input_vector)
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    if sentiment_label in sentiment_map:
        sentiment_idx = sentiment_map[sentiment_label]
        if prediction_proba[0][sentiment_idx] > 0:
            print(f"Keyword '{keyword}' in emotion '{sentiment_label}' predicted successfully")

            # Use feature_log_prob_ to get the log probability of features
            feature_names = vectorizer.get_feature_names_out()
            top_indices = np.argsort(-model.feature_log_prob_[sentiment_idx])[:10]
            related_words = [feature_names[i] for i in top_indices]
            return related_words
        else:
            print(f"Keywords '{keyword}' prediction failed")
            return []
    else:
        print(f"Failed: {sentiment_label}")
        return []


# User input for keywords and sentiment
input_keywords = input("keywords: ")
input_sentiment = input("Emotion target: (negative, neutral, positive) ")
related_keywords = reverse_inference(model, vectorizer, input_keywords, input_sentiment)
print(f"Related with '{input_keywords}' keywords are:", related_keywords)
