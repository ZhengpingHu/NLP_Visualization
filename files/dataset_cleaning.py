import numpy as np
import pandas as pd

commentsLink_17Jan = "../dataset/CommentsJan2017.csv"
articlesLink_17Jan = "../dataset/ArticlesJan2017.csv"

article = pd.read_csv(articlesLink_17Jan)
print(article.shape)

print(article.head())