import os
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

nlp = spacy.load("en_core_web_sm")

dataset_dir = "../dataset/"
output_file = os.path.join(dataset_dir, "ProcessedCommentsJan2017.csv")

analyzer = SentimentIntensityAnalyzer()

def sentiment_analysis(comment):
    sentiment = analyzer.polarity_scores(comment)
    return [sentiment['neg'], sentiment['neu'], sentiment['pos']]

def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return ' '.join(nouns)

def process_in_parallel(func, data, num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 2)
    with Pool(processes=num_processes) as pool:
        result = list(tqdm(pool.imap(func, data), total=len(data), desc="Processing in parallel"))
    return result

if __name__ == '__main__':

    article_file = "ArticlesJan2017.csv"
    comment_file = "CommentsJan2017.csv"

    articles = pd.read_csv(os.path.join(dataset_dir, article_file))
    comments = pd.read_csv(os.path.join(dataset_dir, comment_file), low_memory=False)

    art_sub = articles[['articleID', 'headline', 'keywords', 'sectionName', 'articleWordCount', 'pubDate']].copy()
    com_sub = comments[['articleID', 'commentBody', 'approveDate']]

    comment_count_per_article = com_sub.groupby('articleID').size().to_dict()
    art_sub.loc[:, 'commentCount'] = art_sub['articleID'].map(comment_count_per_article).fillna(0).astype(int)

    comments_text = com_sub['commentBody'].tolist()

    sentiment_vectors = process_in_parallel(sentiment_analysis, comments_text)

    comments_nouns = process_in_parallel(extract_nouns, comments_text)

    print("Applying TF-IDF to nouns...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(tqdm(comments_nouns, desc="TF-IDF Transformation"))

    sentiment_df = pd.DataFrame(sentiment_vectors, columns=['neg', 'neu', 'pos'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    combined_df = pd.concat([com_sub[['articleID', 'commentBody']], sentiment_df, tfidf_df], axis=1)

    combined_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
