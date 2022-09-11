
import os
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

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_metric

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.nn import Linear, Parameter
from torch import nn

from src.config import Paths, Datasets, Models
from src.dataset.get_data import RawDataset
from src.utils.wordcloud import create_wordcloud
from src.dataset.preprocessing import Preprocessor, Dataset
from src.utils.augment_queries import get_new_samples_counts, augment_contextual_synonyms, augment_backtranslation, insert_typos, swap_characters, delete_random_charaters

def standard_ml():
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
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        [X], [y], test_size=0.3
    )[0]


    if augment:
        # Augment train data
        train = pd.DataFrame({'X':X_train, 'y':y_train})

        label=0
        aug_context = pd.DataFrame({
            'X': augment_contextual_synonyms(train.loc[train.y==label].iloc[20:40], 'X'),
            'y': label
        })
        aug_backtrans = pd.DataFrame({
            'X': augment_backtranslation(train.loc[train.y==label].iloc[0:20], 'X'),
            'y': label
        })
        aug_typos = pd.DataFrame({
            'X': insert_typos(train.loc[train.y==label], 'X'),
            'y': label
        })
        aug_swap = pd.DataFrame({
            'X': swap_characters(train.loc[train.y==label], 'X'),
            'y': label
        })
        aug_delete = pd.DataFrame({
            'X': delete_random_charaters(train.loc[train.y==label], 'X'),
            'y': label
        })
        train = pd.concat(train, aug_context, aug_backtrans, aug_typos, aug_swap,
                  aug_delete, axis=0)

        label=1
        aug_context = pd.DataFrame({
            'X': augment_contextual_synonyms(train.loc[train.y==label], 'X'),
            'y': label
        })
        aug_backtrans = pd.DataFrame({
            'X': augment_backtranslation(train.loc[train.y==label], 'X'),
            'y': label
        })
        aug_typos = pd.DataFrame({
            'X': insert_typos(train.loc[train.y==label], 'X'),
            'y': label
        })
        aug_swap = pd.DataFrame({
            'X': swap_characters(train.loc[train.y==label], 'X'),
            'y': label
        })
        aug_delete = pd.DataFrame({
            'X': delete_random_charaters(train.loc[train.y==label], 'X'),
            'y': label
        })
        train = pd.concat(train, aug_context, aug_backtrans, aug_typos, aug_swap,
                  aug_delete, axis=0)

        X_train = train['X']
        y_train = train['y']




    # Random Forrest pipeline
    pipe = Pipeline([
        ('tfidf', preprocessor.vector_token_izer),
        ('scaler', Normalizer()),
        ('svd', TruncatedSVD(n_components=50)),
        ('rfc', RandomForestClassifier()),
    ])
    params = {
        'svd__n_components': [30,  50,  100, 150],
        'rfc__n_estimators': [100, 200, 300, 400],
    }
    gridcv = GridSearchCV(pipe, params, n_jobs=-1, cv=5, scoring='f1')
    gridcv.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % gridcv.best_score_)
    print(gridcv.best_params_)
    gridcv.cv_results_

    model = gridcv.best_estimator_

    # eval on test
    y_test_pred = model.predict(X_test)
    confusion_matrix(y_test, y_test_pred)
    print(classification_report(y_test, y_test_pred))

    X_pred = rawdata.X[1]['Body']
    predictions = pd.DataFrame({
        'texts': rawdata.X[1]['Body'],
        'labels': model.predict(X_pred)
    })
    predictions.to_csv('data/predictions.csv')

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
    pass




def tiny_bert():
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
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Prediction
    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(Models.TINY_BERT,
                                                               num_labels=2)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # prepare data
    test = rawdata.data[1]
    X_test_tokenized = preprocessor.transform(test, 'Body')
    # transform data to torch Dataset style
    test_dataset = Dataset(X_test_tokenized)
    # create dataloader
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
    test.Rating_pred.value_counts()
    pass


if __name__=='__main__':
    pass

    import torch
    import timeit
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]

num_labels = len(model.config.id2label)



    testcode = """
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    """
    settings = """
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    """
    inference_time = timeit.repeat(stmt=testcode, setup=settings, number=1, repeat=100)
    np.mean(inference_time)
    np.std(inference_time)
