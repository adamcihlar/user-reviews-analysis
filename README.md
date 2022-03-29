### Phone user reviews analysis

Conducted as a part of project for Brand Management course taken at Inland Norway University of Applied Sciences.
The goal is to discover users' judgment and feelings about Fairphone brand and their main product to help to assess the company's current position in the market and opportunities for further growth.

Full pipeline should be applicable to any other brand that can be found on trustpilot.com and/or gsmarena.com. 

* data.py - scrape data from trustpilot and gsmarena + wordcloud
* preprocessing.py - get data ready for models - tokenizing/tfidf, adjust labels
* main.py - train model to classify negative and positive reviews, predict the sentiment of gsmarena reviews
* further work: use ratings in EDA, discover topics in reviews, deal with ambiguous word meanings in reviews ("camera", "price" - bad or good?) using the ratings
