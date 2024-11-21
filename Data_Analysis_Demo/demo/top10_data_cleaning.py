import pandas as pd
import re
from glob import glob


def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)


articles_files = sorted(glob("../dataset/Articles*.csv"))
comments_files = sorted(glob("../dataset/Comments*.csv"))

articles_columns = ['articleID', 'headline', 'articleWordCount']
comments_columns = ['articleID', 'commentBody']

for articles_file in articles_files:
    article_month = re.search(r'Articles(\w+\d{4})\.csv', articles_file).group(1)
    matching_comments_file = [f for f in comments_files if article_month in f]

    if matching_comments_file:
        comments_file = matching_comments_file[0]
        articles_df = pd.read_csv(articles_file, usecols=articles_columns)
        comments_df = pd.read_csv(comments_file, usecols=comments_columns)

        articles_df['hlwordcount'] = articles_df['headline'].apply(count_words)
        merged_df = comments_df.merge(articles_df, on='articleID', how='inner')

        comment_counts = merged_df.groupby('articleID').size().reset_index(name='commentCount')
        merged_df = merged_df.merge(comment_counts, on='articleID')

        top_articles_ids = (
            merged_df.groupby('articleID')
            .head(1)
            .sort_values(by='commentCount', ascending=False)
            .head(10)['articleID']
        )

        filtered_comments = merged_df[merged_df['articleID'].isin(top_articles_ids)]

        # Reorder columns: article-related first, commentBody last
        reordered_columns = ['articleID', 'headline', 'hlwordcount',
                             'articleWordCount', 'commentCount', 'commentBody']
        filtered_comments = filtered_comments[reordered_columns]

        output_file = f"../dataset/Top10_Comments_{article_month}.csv"
        filtered_comments.to_csv(output_file, index=False)
        print(f"File saved: {output_file}")
