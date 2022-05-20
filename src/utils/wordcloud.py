import os
from typing import List
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def create_wordcloud(df, column_with_text, stopwords: List[str], save_path):
    text = df[column_with_text].str.cat(sep=' ')
    stop_words = stopwords + list(STOPWORDS)

    plt.figure()
    wc = WordCloud(
        stopwords = stop_words,
        background_color="white",
        collocation_threshold = 3,
        max_words=30
    ).generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    plt.savefig(save_path)
    pass
