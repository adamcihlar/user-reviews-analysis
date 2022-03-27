
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

class Dataset(torch.utils.data.Dataset):
    '''Keeps data in format accepted by the torch DataLoader'''

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class Preprocessor:
    '''Converts raw data to data in format accepted by torch Dataset class or
    Scikit-learn models.

    Receives a tokenizer instance to convert the text to vectors.
    Method for tokenizing, converting the classes to binary problem, splitting
    the data to train and val/test samples, TF-IDF encoding.
    Initialization arg must be changed if using not BERT based tokenizer.
    '''
    def __init__(
        self,
        tokenizer = BertTokenizer.from_pretrained(Models.TINY_BERT),
        split_size = 0.2
        ):
        self.tokenizer = tokenizer
        self.split_size = split_size

    def tokenize(self, df, column_to_tokenize: str, padding=True,
                 truncation=True, max_length=512, **kwargs):
        X = list(df[column_to_tokenize])
        X_tokenized = self.tokenizer(X, padding=padding, truncation=truncation,
                                     max_length=max_length, **kwargs)
        return X_tokenized

    def split_data(self, X, y, test_size=None):
        if test_size is None:
            test_size = self.split_size
        return [train_test_split(X[i], y[i], test_size=self.split_size) for i in range(len(X))]

    def shift_labels(self, y):
        y_shifted = []
        for i in range(len(y)):
            y_shifted.append(y[i] - min(y[i]))
        return y_shifted

    def binarize_labels(self, y, positive_treshold=3):
        y_bin = []
        for i in range(len(y)):
            y_bin.append((y[i] >= positive_treshold).astype('int'))
        return y_bin



if __name__=='__main__':

    raw_dataset = RawDataset()
    raw_dataset.load()
    raw_dataset.X
    raw_dataset.y

    ### CONTINUTE
    # add method converting labels to positive x negative to Preprocessor
    # add method for TF-IDF vectorizing
    # add dim reduction to Preprocessor?
    preprocessor = Preprocessor()
    y = preprocessor.shift_labels(raw_dataset.y)
    y = preprocessor.binarize_labels(raw_dataset.y)
    X_train, X_val, y_train, y_val = preprocessor.split_data(raw_dataset.X, y)[0]
    X_train_tok = preprocessor.tokenize(X_train, 'Body')
    X_val_tok = preprocessor.tokenize(X_val, 'Body')

    # Define pretrained tokenizer and model
    model_name = "prajjwal1/bert-tiny"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Preprocess data
    X = list(train["Body"])
    y = list(train["Rating"]-1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    # transform data to torch Dataset style
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=4)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.config.hidden_dropout_prob = 0.3

    # specify training details
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 6
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler(
        name="cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps*0.15,
        num_training_steps=num_training_steps,
    )

    # gpu if available, else cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # training loop
    val_metric = load_metric("accuracy")
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

        val_metric_progress.append(val_metric.compute()['accuracy'])
        print('Validation accuracy:', val_metric_progress[len(val_metric_progress)-1])

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=5)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # prepare data
    X_test = list(test["Body"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
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
