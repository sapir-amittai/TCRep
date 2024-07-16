import datetime
import random
import numpy as np
from datasets import Dataset
import pandas as pd
from embedding_esm2 import run_model
from loguru import logger
import os


def CLTransformer(transformer, Y):
    """
    augment sequences using sequences from the same class
    :param transformer: dictionary {label: [sequences...]}
    :param Y: ndarray of shape (N,)
    :return:
    """
    return np.array(random.choice(transformer[y]) for y in Y)


def build_datasets(tokenizer, train_seqs, validation_seqs, test_seqs, train_labels, validation_labels, test_labels):
    """
    :param tokenizer: sequence tokenizer
    :param train_seqs: list
    :param train_labels: list
    :param validation_seqs: list
    :param validation_labels: list
    :param test_seqs: list
    :param test_labels: list
    :return: datasets.Dataset objects (train, validation, test)
    """

    train_tokenized, validation_tokenized, test_tokenized = \
        tokenizer(train_seqs), tokenizer(validation_seqs), tokenizer(test_seqs)
    train_dataset, validation_dataset, test_dataset = Dataset.from_dict(train_tokenized), \
                                                      Dataset.from_dict(validation_tokenized), \
                                                      Dataset.from_dict(test_tokenized)
    train_dataset = train_dataset.add_column("labels", train_labels)
    validation_dataset = validation_dataset.add_column("labels", validation_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    return train_dataset, validation_dataset, test_dataset


# def create_sequence_embeddings(sequences, max_seq_len, embedding_model, batch_converter, device, batch_size=1024):
#     """
#     :param sequences: sequence
#     :param max_seq_len:
#     :param embedding_model:
#     :param batch_converter:
#     :param device:
#     :param batch_size:
#     :return: np array of shape(n_sequences, max_seq_length, esm_features_num)
#     """
#     embedded_data = np.zeros((len(sequences), max_seq_len, ESM_FEATURE_NUM))
#     n_batches = (len(sequences) // batch_size) + 1

def get_df_with_esm_results(model, alphabet, sequences, num_layers=[27, 33], batch_size = 10, folder_to_save='', when_save_df=None, break_after_save=False, start_batch=0, mean_embedding=True, prefix_filepath_save=''):
    assert (when_save_df == None) or (when_save_df % batch_size == 0), f"batch_size needs to devide in when_save_df but batch_size is {batch_size} and when_save_df is {when_save_df}"

    os.makedirs(folder_to_save, exist_ok=True)
    base_columns = ['seq', 'esm2_layer', 'idx_to_embedd']
    df = None
    columns = ""

    for i in range(start_batch, len(sequences), batch_size):    
        logger.info(f"running batch {i}")
        batch_seq = sequences[i:i+batch_size]
        # sequence_representations = run_model(model, alphabet, num_layers, batch_seq, mean_embedding=True)
        sequence_representations = run_model(model, alphabet, num_layers, batch_seq, mean_embedding)

        if df is None:
            num_embeddings = len(sequence_representations[list(sequence_representations.keys())[0]][0][0])
            columns = base_columns + [f"embedding_{i}" for i in range(num_embeddings)]
            df = pd.DataFrame(columns=columns)

        data_rows = []
        for esm2_layer in sequence_representations.keys():
            for idx_seq, seq in enumerate(batch_seq):
                for idx_to_embedd, embedding in enumerate(sequence_representations[esm2_layer][idx_seq]):
                    embedding_lst = embedding.tolist()
                    embedding_lst.insert(0, idx_to_embedd)
                    embedding_lst.insert(0, esm2_layer)
                    embedding_lst.insert(0, seq)
                    data_rows.append(embedding_lst)

        new_rows_df = pd.DataFrame(data_rows, columns=df.columns)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        data_rows = []

        if when_save_df is not None:
            if i % when_save_df == 0 and i != 0:
                save_df(batch_size, folder_to_save, df, i, prefix_filepath_save)
                df = pd.DataFrame(columns=columns)
                if break_after_save:
                    break

    if when_save_df is None:
        save_df(batch_size, folder_to_save, df, i, prefix_filepath_save)                

def save_df(batch_size, folder_to_save, df, i, prefix_filepath_save):
    save_path = os.path.join(folder_to_save, f"{prefix_filepath_save}until_seq_{i+batch_size}.csv")
    logger.info(f"savind df(the data) from the last saving until {i+batch_size}")
    logger.info(f"sanivg file to path {save_path}")
    df.to_csv(save_path, index=False)
            
