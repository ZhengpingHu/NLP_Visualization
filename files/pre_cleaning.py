import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import spacy

nlp = spacy.load("en_core_web_sm")

dataset_dir = "../dataset/"
output_file = os.path.join(dataset_dir, "ProcessedCommentsAll.csv")

analyzer = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(max_features=500)

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

def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return ' '.join(nouns)

def sentiment_vector(comment):
    sentiment = analyzer.polarity_scores(comment)
    return [sentiment['neg'], sentiment['neu'], sentiment['pos']]

def process_sentiment_batch(batch_comments):
    return [sentiment_vector(comment) for comment in batch_comments]

def parallel_sentiment_analysis(comments_text, num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 2)

    batch_size = len(comments_text) // num_processes
    batches = [comments_text[i:i + batch_size] for i in range(0, len(comments_text), batch_size)]

    with Pool(processes=num_processes) as pool:
        sentiment_batches = list(tqdm(pool.imap(process_sentiment_batch, batches), total=len(batches),
                                      desc="Processing Sentiments in Parallel"))

    all_sentiments = [item for sublist in sentiment_batches for item in sublist]
    return all_sentiments

all_comments = []
all_sentiments = []
all_nouns = []

for article_file, comment_file in zip(article_files, comment_files):
    articles = pd.read_csv(os.path.join(dataset_dir, article_file))
    comments = pd.read_csv(os.path.join(dataset_dir, comment_file), low_memory=False)

    art_sub = articles[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']].copy()
    com_sub = comments[['articleID', 'commentBody', 'approveDate']]

    comment_count_per_article = com_sub.groupby('articleID').size().to_dict()
    art_sub.loc[:, 'commentCount'] = art_sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)

    comments_text = com_sub['commentBody'].tolist()
    all_comments.extend(comments_text)

    sentiments = parallel_sentiment_analysis(comments_text)
    all_sentiments.extend(sentiments)

    nouns_batch = [extract_nouns(comment) for comment in comments_text]
    all_nouns.extend(nouns_batch)

print("Applying TF-IDF on extracted nouns...")
tfidf_matrix = vectorizer.fit_transform(tqdm(all_nouns, desc="TF-IDF Transformation on Nouns"))

sentiment_df = pd.DataFrame(all_sentiments, columns=['neg', 'neu', 'pos'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
combined_df = pd.concat([sentiment_df, tfidf_df], axis=1)

combined_df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
