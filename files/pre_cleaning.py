import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

dataset_dir = "../dataset/"
output_file = os.path.join(dataset_dir, "CombinedProcessedComments.csv")

article_files = [
    "ArticlesJan2017.csv", "ArticlesFeb2017.csv", "ArticlesMarch2017.csv", "ArticlesApril2017.csv",
    "ArticlesMay2017.csv",
    "ArticlesJan2018.csv", "ArticlesFeb2018.csv", "ArticlesMarch2018.csv", "ArticlesApril2018.csv"
]

comment_files = [
    "CommentsJan2017.csv", "CommentsFeb2017.csv", "CommentsMarch2017.csv", "CommentsApril2017.csv",
    "CommentsMay2017.csv",
    "CommentsJan2018.csv", "CommentsFeb2018.csv", "CommentsMarch2018.csv", "CommentsApril2018.csv"
]

analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(max_features=500)


all_comments = []
all_sentiment_vectors = []

for article_file, comment_file in tqdm(zip(article_files, comment_files), total=len(article_files),
                                       desc="Processing Files"):
    articles = pd.read_csv(os.path.join(dataset_dir, article_file))
    comments = pd.read_csv(os.path.join(dataset_dir, comment_file), low_memory=False)
    art_sub = articles[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']].copy()
    com_sub = comments[['articleID', 'commentBody', 'approveDate']]
    comment_count_per_article = com_sub.groupby('articleID').size().to_dict()
    art_sub.loc[:, 'commentCount'] = art_sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)
    comments_text = com_sub['commentBody'].tolist()
    all_comments.extend(comments_text)

    for comment in tqdm(comments_text, desc=f"Processing Comments for {comment_file}", leave=False):
        sentiment = analyzer.polarity_scores(comment)
        all_sentiment_vectors.append([sentiment['neg'], sentiment['neu'], sentiment['pos']])

print("Applying TF-IDF...")
tfidf_matrix = vectorizer.fit_transform(tqdm(all_comments, desc="TF-IDF Transformation"))

sentiment_df = pd.DataFrame(all_sentiment_vectors, columns=['neg', 'neu', 'pos'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

combined_df = pd.concat([sentiment_df, tfidf_df], axis=1)

combined_df.to_csv(output_file, index=False)
print(f"Complete, dataset saved at: {output_file}")
