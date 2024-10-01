import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

articlesLink_17Jan = "../dataset/ArticlesJan2017.csv"
commentsLink_17Jan = "../dataset/CommentsJan2017.csv"


articles_17Jan = pd.read_csv(articlesLink_17Jan)
art_17Jan_Sub = articles_17Jan[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']]


comments_17Jan = pd.read_csv(commentsLink_17Jan, low_memory=False)
com_17Jan_Sub = comments_17Jan[['articleID', 'commentBody', 'approveDate']]
print(art_17Jan_Sub.shape)
print(com_17Jan_Sub.shape)

comment_count_per_article = com_17Jan_Sub.groupby('articleID').size().to_dict()


art_17Jan_Sub.loc[:, 'commentCount'] = art_17Jan_Sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)


sorted_articles = art_17Jan_Sub.sort_values(by='commentCount', ascending=False)


top_article_features = sorted_articles[['headline', 'keywords', 'commentCount']]


print(top_article_features)

plt.figure(figsize=(10, 6))
plt.bar(top_article_features['headline'], top_article_features['commentCount'], color='blue')
plt.title('Top Article Comment Count')
plt.xlabel('Headline')
plt.ylabel('Comment Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()