
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

from torch.nn import Linear, Parameter
from torch import nn

from src.config import Paths, Datasets, Models
from src.dataset.get_data import RawDataset
from src.utils.wordcloud import create_wordcloud
from src.dataset.preprocessing import Preprocessor, Dataset

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

    # Random Forrest pipeline
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


    ### Embeddings approach - Tiny BERT
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_scheduler
    from tqdm.auto import tqdm
    from datasets import load_metric

    from transformers import BertTokenizer, BertForSequenceClassification
    from transformers import AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup

    from src.data import RawDataset
    from src.config import Models

    # Load data
    rawdata = RawDataset()
    #rawdata.download_and_save()
    rawdata.load()
    X = rawdata.X[0]['Body']
    y = rawdata.y[0]

    # Preprocess data
    preprocessor = Preprocessor(method='tokenize')
    y = preprocessor.binarize_labels(y)
    X_train, X_val, y_train, y_val = preprocessor.split_data([rawdata.X[0]], [y])[0]
    X_train_tok = preprocessor.fit_transform(X_train, 'Body')
    X_val_tok = preprocessor.transform(X_val, 'Body')

    # transform data to torch Dataset style
    train_dataset = Dataset(X_train_tok, y_train)
    val_dataset = Dataset(X_val_tok, y_val)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=4)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(Models.TINY_BERT, num_labels=2)
    model.config.hidden_dropout_prob = 0.3

    # specify training details
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps*0.15,
        num_training_steps=num_training_steps,
    )

    # gpu if available, else cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # training loop
    val_metric = load_metric("f1") #accuracy
    val_metric_progress = []

    progress_bar = tqdm(range(num_training_steps))
    counter = 0

    for epoch in range(num_epochs):

        model.eval()
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_metric.add_batch(predictions=predictions, references=batch["labels"])

        val_metric_progress.append(val_metric.compute()['f1'])
        print('Validation metric:', val_metric_progress[len(val_metric_progress)-1])

        # training
        model.train()
        loss_progress = []
        for batch in train_dataloader:
            counter += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            loss_progress.append(loss.item())
            if counter % 5 == 0:
                mean_loss = np.mean(np.array(loss_progress))
                print('Training loss:', mean_loss)
                loss_progress = []


        # validation
        model.eval()
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_metric.add_batch(predictions=predictions, references=batch["labels"])

        val_metric_progress.append(val_metric.compute()['f1'])
        print('Validation metric:', val_metric_progress[len(val_metric_progress)-1])

    # inspect validation errors
    val_predictions = []
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        val_predictions += list(predictions.cpu().detach().numpy())

    confusion_matrix(y_val, val_predictions)


    # save the model
    save_path = 'models/tiny_bert_classifier'
    torch.save(model.state_dict(), save_path)


    # Prediction
    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(Models.TINY_BERT,
                                                               num_labels=2)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # prepare data
    X_test = rawdata.data[1]
    X_test_tokenized = preprocessor.transform(X_test, 'Body')
    # transform data to torch Dataset style
    test_dataset = Dataset(X_test_tokenized)
    # create dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    test_predictions = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        test_predictions += list(predictions.cpu().detach().numpy())

    test['Rating_pred'] = test_predictions

    # Get embeddings
    # cut the last classifying layer
    # get the embeddings
    # cluser
    # plot in 2D
    # extract topics















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
