import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer


dataset_dir = "../dataset/"

article_files = [f for f in os.listdir(dataset_dir) if f.startswith('Articles') and f.endswith('.csv')]
comment_files = [f for f in os.listdir(dataset_dir) if f.startswith('Comments') and f.endswith('.csv')]


def extract_year_month(file_name):
    year_month = file_name.split('Articles')[1].split('.csv')[0] if 'Articles' in file_name else \
        file_name.split('Comments')[1].split('.csv')[0]
    return year_month

all_headlines = []

for article_file in article_files:
    year_month = extract_year_month(article_file)
    comment_file = [f for f in comment_files if year_month in f]

    if comment_file:
        articles = pd.read_csv(os.path.join(dataset_dir, article_file))
        comments = pd.read_csv(os.path.join(dataset_dir, comment_file[0]), low_memory=False)

        art_sub = articles[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']].copy()
        com_sub = comments[['articleID', 'commentBody', 'approveDate']]

        comment_count_per_article = com_sub.groupby('articleID').size().to_dict()

        art_sub.loc[:, 'commentCount'] = art_sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)

        sorted_articles = art_sub.sort_values(by='commentCount', ascending=False)
        top_20_articles = sorted_articles[['headline', 'keywords', 'commentCount']].head(20)

        all_headlines.extend(top_20_articles['headline'].tolist())

        print(f"Top 20 Articles for {year_month}:")
        print(top_20_articles)

        plt.figure(figsize=(20, 10))
        plt.bar(top_20_articles['headline'], top_20_articles['commentCount'], color='blue')
        plt.title(f'Top 20 Articles Comment Count - {year_month}')
        plt.xlabel('Headline')
        plt.ylabel('Comment Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

print("All Headline List:")
print(all_headlines)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_headlines)

feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

print("\nTF-IDF Matrix for Headlines:")
print(df)

for idx, row in df.iterrows():
    top_keywords = row.sort_values(ascending=False).head(10)
    print(f"\nHeadline {idx + 1} Top Keywords:")
    print(top_keywords)
