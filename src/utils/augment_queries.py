
from typing import List

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac


def get_new_samples_counts(
    df: pd.DataFrame,
    total_new_samples_count: int,
    original_queries_weights: List[int] = [1]
):
    '''
    Creates np.ndarray of length == len(df) with number of new samples that
    should be generated for the given original query.
    If df % total_new_samples_count != 0, the remaining samples are distributed
    randomly.
    If original_queries_weights is passed, the count of new samples for each
    query is weighted by the corresponding weight.
    '''
    original_queries_weights = np.repeat(
        original_queries_weights,
        repeats=(df.shape[0]/len(original_queries_weights))
    )

    new_samples_counts = ((
        total_new_samples_count / np.sum(original_queries_weights)
    ) * original_queries_weights).round(decimals=0)

    remaining = total_new_samples_count - np.sum(new_samples_counts)
    new_samples_addition = np.concatenate((
        np.repeat(1, remaining),
        np.repeat(0, df.shape[0] - remaining)
    ))
    new_samples_counts += np.random.permutation(new_samples_addition)

    return new_samples_counts


def augment_contextual_synonyms(
    one_class_df: pd.DataFrame,
    query_column: str='query_text',
    model: str='bert-base-uncased',
    stopwords=nltk.corpus.stopwords.words('english'),
    top_k_closest_words = 10,
    n_new_samples: np.ndarray = 1
):
    '''
    Creates n_new_samples for every provided query by replacing words in the
    query with contextually similar words based on contextual word embeddings.

    one_class_df: pd.DataFrame
        should contain only one class for easier manipulation
    query_column: str
        column with queries to augment
    stopwords: list[str]
        these words will not be augmented
    top_k_closest_words: int
        how many contextually similar words to consider for replacement
    n_new_samples: np.array
        either int if n_new_samples is the same for all queries
        or np.array(int) of length equal to n rows of one_class_df

    Returns:
        list with new samples
    '''

    one_class_df['n_new_samples'] = n_new_samples

    df_1 = one_class_df.loc[one_class_df.n_new_samples==1]
    df_n = one_class_df.loc[one_class_df.n_new_samples>1]

    # TODO consider adding mck dictionary to stopwords
    aug = naw.ContextualWordEmbsAug(
        model_path=model,
        stopwords=stopwords,
        top_k=top_k_closest_words
    )

    # augmentation in batches if n_new_samples==1
    new_texts = []
    texts = list(df_1[query_column])
    new_texts += aug.augment(texts)

    # augmentation separately for every query if n_new_samples>1
    for index, row in df_n.iterrows():
        new_texts += aug.augment(row[query_column], n=row['n_new_samples'])

    return new_texts


def augment_backtranslation(
    one_class_df: pd.DataFrame,
    query_column: str='query_text',
    from_model: str='facebook/wmt19-en-de',
    to_model: str='facebook/wmt19-de-en',
):
    aug = naw.BackTranslationAug(
        from_model_name=from_model,
        to_model_name=to_model
    )
    new_texts = aug.augment(list(one_class_df[query_column]))
    return new_texts


def insert_typos(
    one_class_df: pd.DataFrame,
    query_column: str='query_text',
    n_new_samples: np.ndarray = 1,
):
    '''
    Creates n_new_samples for every provided query by replacing characters in the
    query with characters close on the keybord.

    one_class_df: pd.DataFrame
        should contain only one class for easier manipulation
    query_column: str
        column with queries to augment
    n_new_samples: np.array
        either int if n_new_samples is the same for all queries
        or np.array(int) of length equal to n rows of one_class_df

    Returns:
        list with new samples
    '''

    one_class_df['n_new_samples'] = n_new_samples

    df_1 = one_class_df.loc[one_class_df.n_new_samples==1]
    df_n = one_class_df.loc[one_class_df.n_new_samples>1]

    aug = nac.KeyboardAug(aug_char_max=1, aug_word_max=1, include_upper_case=False)

    # augmentation in batches if n_new_samples==1
    new_texts = []
    texts = list(df_1[query_column])
    new_texts += aug.augment(texts)

    # augmentation separately for every query if n_new_samples>1
    for index, row in df_n.iterrows():
        new_texts += aug.augment(row[query_column], n=row['n_new_samples'])

    return new_texts


def swap_characters(
    one_class_df: pd.DataFrame,
    query_column: str='query_text',
    n_new_samples: np.ndarray = 1,
):
    '''
    Creates n_new_samples for every provided query by swaping chars in the query.

    one_class_df: pd.DataFrame
        should contain only one class for easier manipulation
    query_column: str
        column with queries to augment
    n_new_samples: np.array
        either int if n_new_samples is the same for all queries
        or np.array(int) of length equal to n rows of one_class_df

    Returns:
        list with new samples
    '''

    one_class_df['n_new_samples'] = n_new_samples

    df_1 = one_class_df.loc[one_class_df.n_new_samples==1]
    df_n = one_class_df.loc[one_class_df.n_new_samples>1]

    aug = nac.RandomCharAug(action="swap", aug_char_max=1, include_upper_case=False)

    # augmentation in batches if n_new_samples==1
    new_texts = []
    texts = list(df_1[query_column])
    new_texts += aug.augment(texts)

    # augmentation separately for every query if n_new_samples>1
    for index, row in df_n.iterrows():
        new_texts += aug.augment(row[query_column], n=row['n_new_samples'])

    return new_texts


def delete_random_charaters(
    one_class_df: pd.DataFrame,
    query_column: str='query_text',
    n_new_samples: np.ndarray = 1,
):
    '''
    Creates n_new_samples for every provided query by deleting char in the query.

    one_class_df: pd.DataFrame
        should contain only one class for easier manipulation
    query_column: str
        column with queries to augment
    n_new_samples: np.array
        either int if n_new_samples is the same for all queries
        or np.array(int) of length equal to n rows of one_class_df

    Returns:
        list with new samples
    '''

    one_class_df['n_new_samples'] = n_new_samples

    df_1 = one_class_df.loc[one_class_df.n_new_samples==1]
    df_n = one_class_df.loc[one_class_df.n_new_samples>1]

    aug = nac.RandomCharAug(action="delete", aug_char_max=2, include_upper_case=False)

    # augmentation in batches if n_new_samples==1
    new_texts = []
    texts = list(df_1[query_column])
    new_texts += aug.augment(texts)

    # augmentation separately for every query if n_new_samples>1
    for index, row in df_n.iterrows():
        new_texts += aug.augment(row[query_column], n=row['n_new_samples'])

    return new_texts
