
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

# from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
# from transformers import EarlyStoppingCallback

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
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

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def load_train(path):
    train = pd.read_csv(path)
    train = train[['Body', 'Rating']]
    return train

def load_test():
    rev3 = pd.read_csv('data/rev_3.csv')
    rev3p = pd.read_csv('data/rev_3p.csv')
    rev4 = pd.read_csv('data/rev_4.csv')
    test = pd.concat([rev3, rev3p, rev4])
    test = test[['Body']]
    return test

if __name__=='__main__':

    train = load_train('data/reviews.csv')
    test = load_test()

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
