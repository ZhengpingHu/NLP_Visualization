import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import spacy


nlp = spacy.load("en_core_web_sm")


dataset_dir = "../dataset/"


articles_file = os.path.join(dataset_dir, "ArticlesJan2017.csv")
comments_file = os.path.join(dataset_dir, "CommentsJan2017.csv")


articles = pd.read_csv(articles_file)
comments = pd.read_csv(comments_file, low_memory=False)


art_sub = articles[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']].copy()
com_sub = comments[['articleID', 'commentBody', 'approveDate']]

comment_count_per_article = com_sub.groupby('articleID').size().to_dict()

art_sub.loc[:, 'commentCount'] = art_sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)

top_article = art_sub.sort_values(by='commentCount', ascending=False).head(1)

top_article_file = "top_article_Jan2017.csv"
top_article.to_csv(top_article_file, index=False)

print(f"Top article for Jan 2017 saved as {top_article_file}")

top_article_id = top_article['articleID'].values[0]
top_article_comments = com_sub[com_sub['articleID'] == top_article_id]


headline = top_article['headline'].values[0]
comments_text = " ".join(top_article_comments['commentBody'].tolist())


text_data = headline + " " + comments_text

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform([text_data])


feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)


def filter_emotional_words(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]


emotional_words = filter_emotional_words(" ".join(feature_names))


wordcloud_text = " ".join(emotional_words)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Emotional Word Cloud for Top Article (Jan 2017)')
plt.axis('off')
plt.show()
