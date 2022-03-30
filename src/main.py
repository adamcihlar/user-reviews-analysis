
import numpy as np
import pandas as pd

# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from wordcloud import STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix

from src.config import Paths, Datasets, Models
from src.data import RawDataset, create_wordcloud
from src.preprocessing import Preprocessor, Dataset

if __name__=='__main__':

    ### Standard sklearn pipeline to try out few standard ML models with
    # gridsearch

    # Load data
    rawdata = RawDataset()
    #rawdata.download_and_save()
    rawdata.load()
    X = rawdata.X[0]['Body']
    y = rawdata.y[0]

    # Preprocess data
    preprocessor = Preprocessor(method='vectorize',
                                vector_token_izer=TfidfVectorizer())
    y = preprocessor.binarize_labels(y)

    # Random Forrest Pipeline
    pipe = Pipeline([
        ('tfidf', preprocessor.vector_token_izer),
        ('scaler', Normalizer()),
        ('svd', TruncatedSVD(n_components=50)),
        ('rfc', RandomForestClassifier()),
    ])
    params = {
        'svd__n_components': [30, 40, 50, 70, 100],
        'rfc__n_estimators': [100, 200, 300, 400],
    }
    gridcv = GridSearchCV(pipe, params, n_jobs=-1, cv=5, scoring='f1')
    gridcv.fit(X, y)
    print("Best parameter (CV score=%0.3f):" % gridcv.best_score_)
    print(gridcv.best_params_)
    gridcv.cv_results_
    # I really should create a test to compare models

    model = gridcv.best_estimator_
    # save the model?
    X_test = rawdata.X[1]['Body']
    predictions = pd.DataFrame({
        'texts': rawdata.X[1]['Body'],
        'labels': model.predict(X_test)
    })
    predictions.to_excel('data/predictions.xls')

    truth = pd.DataFrame({
        'texts': rawdata.X[0]['Body'],
        'labels': y
    })

    labeled_dataset = pd.concat([truth, predictions])
    positive_reviews = labeled_dataset.loc[labeled_dataset['labels']==1]
    negative_reviews = labeled_dataset.loc[labeled_dataset['labels']==0]

    stop_words = ["Fairphone", "phone", "will", "back", "still",'one', 'now',
                  'month', 'year', 'make', 'bought', 'got', 'week', 'day',
                  'buy', 'want', 'company', 'call','use', 'really', 'lot',
                  'jack', 'months']
    create_wordcloud(positive_reviews, 'texts', stop_words,
                     'assets/wc_positive.png')
    create_wordcloud(negative_reviews, 'texts', stop_words,
                     'assets/wc_negative.png')


    ### animated plot
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    rawdata.data[0]
    fig, ax = plt.subplots()

    rawdata.data[0]['Quarter'] = pd.to_datetime(rawdata.data[0]['Date']).dt.to_period('Q')

    x = rawdata.data[0]['Rating'].expanding().mean()
    y = rawdata.data[0]['Date']

#     y = rawdata.data[0].groupby('Quarter').mean()
#     x = rawdata.data[0]['Quarter'].unique()
    line, = ax.plot(x, y)


    def animate(i):
        line.set_ydata(np.sin(x + i / 50))  # update the data.
        return line,


    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, save_count=50)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()
