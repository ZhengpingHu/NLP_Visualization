import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

link = "../dataset/ProcessedCommentsAll.csv"
df = pd.read_csv(link)

df['keywords'] = df['keywords'].fillna('')

df['sentiment_score'] = df[['neg', 'neu', 'pos']].dot([0.5, 0, -0.5])
df = df[['sentiment_score', 'keywords']]

tfidf = TfidfVectorizer(max_features=500)
keywords_sparse = tfidf.fit_transform(df['keywords']).tocoo()

features = np.hstack([df[['sentiment_score']].values, keywords_sparse.toarray()])

np.savez_compressed("../dataset/ProcessedCommentsGNN.npz", features=features)
print("Dataset saved as npz structure.")
