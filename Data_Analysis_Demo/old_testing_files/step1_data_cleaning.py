import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import re
import spacy
from multiprocessing import Pool, cpu_count

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def sentiment_scores(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment["neg"], sentiment["neu"], sentiment["pos"]

def extract_meaningful_words(text):
    doc = nlp(text)
    allowed_pos_tags = {"NOUN", "VERB", "ADJ", "ADV", "X"}
    return " ".join([token.text for token in doc if token.pos_ in allowed_pos_tags])

def process_comment(comment):
    cleaned_text = clean_text(comment['commentBody'])
    neg, neu, pos = sentiment_scores(cleaned_text)
    keywords = extract_meaningful_words(cleaned_text)
    return comment['articleID'], cleaned_text, neg, neu, pos, keywords

def process_comments_in_parallel(data, num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 2)

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap(process_comment, data.to_dict('records')), total=len(data), desc="Processing Comments"))

    processed_data = pd.DataFrame(results,
                                  columns=["articleID", "cleaned_commentBody", "neg", "neu", "pos", "keywords"])
    return processed_data


def main():
    dataset_dir = "../dataset/"
    comment_files = [
        "CommentsApril2017.csv", "CommentsApril2018.csv", "CommentsFeb2017.csv",
        "CommentsFeb2018.csv", "CommentsJan2017.csv", "CommentsJan2018.csv",
        "CommentsMarch2017.csv", "CommentsMarch2018.csv", "CommentsMay2017.csv"
    ]

    all_data = []

    for file_name in comment_files:
        print(f"Processing {file_name}...")
        data = pd.read_csv(os.path.join(dataset_dir, file_name), usecols=["commentBody", "articleID"])

        processed_data = process_comments_in_parallel(data)

        all_data.append(processed_data)

    final_data = pd.concat(all_data, ignore_index=True)
    output_file = os.path.join(dataset_dir, "ProcessedCommentsAll.csv")
    final_data.to_csv(output_file, index=False)
    print(f"All data processed and saved to {output_file}")


if __name__ == "__main__":
    main()
