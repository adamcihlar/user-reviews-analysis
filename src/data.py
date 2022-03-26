
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time
import lxml
from typing import List, Tuple
import re
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

from src.config import Paths

class RawDataset:
    '''Responsible for loading or downloading the raw data.

    Keeps all data in original format as defined in scraping
    functions. Further preprocessing is not in scope of this class, to be done
    based on the next step.
    '''
    def __init__(
        self,
        paths: List[str] = [
            path.join(Paths.DATA, 'trustpilot_data.csv'),
            path.join(Paths.DATA, 'gsmarena_data.csv'),
        ],
        url_paths: List[Tuple[str, int]] = [
        ('https://www.trustpilot.com/review/fairphone.com', 10),
        ('https://www.gsmarena.com/fairphone_3-reviews-10397.php', 1),
        ('https://www.gsmarena.com/fairphone_3+-reviews-10069.php', 5),
        ('https://www.gsmarena.com/fairphone_4-reviews-11136.php', 10),
        ]
    ):
        self.paths=paths
        self.url_paths=url_paths
        self.data=[]

    def load(self, new_paths: List[str] = None):
        temp_data = []
        if new_paths is not None:
            self.paths = new_paths
        for path_to_file in self.paths:
            if path.isfile(path_to_file):
                temp_data.append(pd.read_csv(path_to_file))
            else:
                print('File: ', path_to_file, ' not found.')

        trust_data = [df for df in temp_data if len(df.columns)==8]
        gsm_data = [df for df in temp_data if len(df.columns)!=8]

        if len(trust_data)!=0:
            trust_df = pd.concat(trust_data)
        if len(gsm_data)!=0:
            gsm_df = pd.concat(gsm_data)

        self.data = [trust_df, gsm_df]


    def download_and_save(self):
        trust_data = []
        gsm_data = []
        for url, n_pages in self.url_paths:
            if re.search('trustpilot.com', url):
                trust_data.append(self._trustpilot_scraper(url, n_pages))
            elif re.search('gsmarena.com', url):
                gsm_data.append(self._gsmarena_scraper(url, n_pages))
        trust_df = pd.concat(trust_data)
        gsm_df = pd.concat(gsm_data)
        trust_df.to_csv(
            path.join(Paths.DATA, 'trustpilot_data.csv')
        )
        gsm_df.to_csv(
            path.join(Paths.DATA, 'gsmarena_data.csv')
        )

        self.data = [trust_df, gsm_df]


    def _trustpilot_scraper(self, PATH: str, n_pages):
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
        rev_df['Link'] = PATH

        return rev_df


    def _gsmarena_scraper(self, PATH: str, n_pages):

        PATH_body = PATH[0:-4]
        PATH_append = PATH[-4:]
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
        rev_df['Link'] = PATH

        return rev_df


if __name__=='__main__':

    raw_dataset = RawDataset()
    #dataset.download_and_save()
    raw_dataset.load()

    text = ''
    nrows = 0
    for df in raw_dataset.data:
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
    plt.savefig(path.join(Paths.ASSETS, 'wordcloud.png'))
