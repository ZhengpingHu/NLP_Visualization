import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

dataset_dir = "../dataset/"

analyzer = SentimentIntensityAnalyzer()

articles_file = os.path.join(dataset_dir, "ArticlesJan2017.csv")
comments_file = os.path.join(dataset_dir, "CommentsJan2017.csv")

articles = pd.read_csv(articles_file)
comments = pd.read_csv(comments_file, low_memory=False)

art_sub = articles[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']].copy()
com_sub = comments[['articleID', 'commentBody', 'approveDate']]

comment_count_per_article = com_sub.groupby('articleID').size().to_dict()

art_sub.loc[:, 'commentCount'] = art_sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)

top_article = art_sub.sort_values(by='commentCount', ascending=False).head(1)

top_article_id = top_article['articleID'].values[0]
top_article_comments = com_sub[com_sub['articleID'] == top_article_id]


comments_text = top_article_comments['commentBody'].tolist()

def sentiment_vector(comment):
    sentiment = analyzer.polarity_scores(comment)
    return np.array([sentiment['neg'], sentiment['neu'], sentiment['pos']])



sentiment_vectors = np.array([sentiment_vector(comment) for comment in comments_text])


neg_values = sentiment_vectors[:, 0]
neu_values = sentiment_vectors[:, 1]
pos_values = sentiment_vectors[:, 2]


fig = go.Figure()


fig.add_trace(go.Scatter3d(
    x=neg_values,
    y=neu_values,
    z=pos_values,
    mode='markers',
    marker=dict(
        size=8,
        color=pos_values,
        colorscale='Viridis',
        opacity=0.8
    ),
    text=comments_text,
))


fig.update_layout(
    scene=dict(
        xaxis_title='Negative Sentiment',
        yaxis_title='Neutral Sentiment',
        zaxis_title='Positive Sentiment',
    ),
    title="Interactive 3D Sentiment Visualization for Comments (Jan 2017)",
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.show()


# The word cloud

all_text = " ".join(comments_text)


# Positive word: green, Neutral word: blue, Negative word: red.
def get_sentiment_color(word):
    sentiment = analyzer.polarity_scores(word)
    if sentiment['compound'] >= 0.05:
        return 'green'
    elif sentiment['compound'] <= -0.05:
        return 'red'
    else:
        return 'blue'


def generate_color_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    def color_func(word, *args, **kwargs):
        return get_sentiment_color(word)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud.recolor(color_func=color_func), interpolation='bilinear')
    plt.axis('off')
    plt.title('Sentiment Word Cloud (Jan 2017)')
    plt.show()

generate_color_wordcloud(all_text)
