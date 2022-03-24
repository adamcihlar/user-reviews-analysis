
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time
import lxml

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def trustpilot_scraper(PATH: str, n_pages):

    #Lists
    body = []
    heading = []
    rating = []
    location = []
    author = []
    date = []

    #Website Load
    page = "{}?page=".format(PATH)
    for page_number in range(1, n_pages+1):
        url = "{x}{y}".format(x = page, y = page_number)
        req = requests.get(url)
        time.sleep(2)
        soup = BeautifulSoup(req.text, 'html.parser')

        print(soup.prettify())

        #initial reviews
        reviews_raw = soup.find("script", id = "__NEXT_DATA__").string
        reviews_raw = json.loads(reviews_raw)
        rev = reviews_raw["props"]["pageProps"]["reviews"]

        #get reviews into df
        for i in range(len(rev)):
            instance = rev[i]
            body_ = instance["text"]
            heading_ = instance["title"]
            rating_ = instance["rating"]
            location_ = instance["consumer"]["countryCode"]
            author_ = instance["consumer"]["displayName"]
            date_ = pd.to_datetime(instance["dates"]["publishedDate"]).strftime("%Y-%m-%d")

            #append to the list
            body.append(body_)
            heading.append(heading_)
            rating.append(rating_)
            location.append(location_)
            author.append(author_)
            date.append(date_)

    df = {
        'Date' : date,
        'Author' : author,
        'Body' : body,
        'Heading' : heading,
        'Rating' : rating,
        'Location' : location
    }

    rev_df = pd.DataFrame(df)
    rev_df.sort_values(by="Date", axis=0, inplace=True, ignore_index=True)
    rev_df.drop_duplicates(subset=["Body"], keep='first', inplace=True)
    rev_df.reset_index(drop=True, inplace=True)

    return rev_df


def gsmarena_scraper(PATH: str, n_pages):

    PATH_body = PATH[0:-4]
    PATH_append = PATH[-4:]
    page_number = 1

    body = []
    dates = []

    #Website Load
    page = "{}p".format(PATH_body)
    for page_number in range(1, n_pages+1):
        url = "{x}{y}{z}".format(x = page, y = page_number, z = PATH_append)
        req = requests.get(url)
        time.sleep(2)
        soup = BeautifulSoup(req.text, 'html.parser')


        #reviews
        for date, review in zip(
            soup.find_all("li", {"class": "upost"}),
            soup.find_all("p", {"class": "uopin"})
        ):
            if not review.find('span', {'class':'uinreply-msg'}):
                dates.append(date.text)
                body.append(review.text)

    df = {
        'Date': dates,
        'Body': body,
    }

    rev_df = pd.DataFrame(df)
    rev_df['Date'] = pd.to_datetime(rev_df['Date'])
    rev_df.sort_values(by="Date", axis=0, inplace=True, ignore_index=True)
    rev_df.drop_duplicates(subset=["Body"], keep='first', inplace=True)
    rev_df.reset_index(drop=True, inplace=True)

    return rev_df



if __name__=='__main__':

#     data_trus = trustpilot_scraper(
#         'https://www.trustpilot.com/review/fairphone.com', 10
#     )
#     data_gsm3 = gsmarena_scraper(
#         'https://www.gsmarena.com/fairphone_3-reviews-10397.php', 1
#     )
#     data_gsm3p = gsmarena_scraper(
#         'https://www.gsmarena.com/fairphone_3+-reviews-10069.php', 5
#     )
#     data_gsm4 = gsmarena_scraper(
#         'https://www.gsmarena.com/fairphone_4-reviews-11136.php', 10
#     )
#     data_trus.to_csv('data/rev_trus.csv')
#     data_gsm3.to_csv('data/rev_3.csv')
#     data_gsm3p.to_csv('data/rev_3p.csv')
#     data_gsm4.to_csv('data/rev_4.csv')

    data_trus = pd.read_csv('data/reviews.csv')
    data_gsm3 = pd.read_csv('data/rev_3.csv')
    data_gsm3p = pd.read_csv('data/rev_3p.csv')
    data_gsm4 = pd.read_csv('data/rev_4.csv')

    text = ''
    nrows = 0
    for df in [data_trus, data_gsm3, data_gsm3p, data_gsm4]:
        text += df.Body.str.cat(sep=' ')
        nrows += len(df)

    stop_words = ["Fairphone", "phone", "will", "back", "still",'one', 'now',
                  'month', 'year', 'make', 'bought', 'got', 'week', 'day',
                  'buy', 'want', 'company', 'call','use', 'really', 'lot',
                  'jack', 'months'] + list(STOPWORDS)
    plt.figure()
    wc = WordCloud(
        stopwords = stop_words,
        background_color="white",
        collocation_threshold = 3,
        max_words=50
    ).generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('assets/wordcloud_test.png')
