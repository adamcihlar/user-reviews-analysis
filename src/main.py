
import numpy as np
import pandas as pd

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
from src.data import RawDataset
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

    model = gridcv.best_estimator_
    X_test = rawdata.X[1]['Body']
    predictions = pd.DataFrame({
        'texts': rawdata.X[1]['Body'],
        'pred_labels': model.predict(X_test)
    })
    predictions.to_excel('data/predictions.xls')
