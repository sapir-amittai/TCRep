import time
from typing import Dict, List
import pandas as pd
from consts import STUDY_ID, SAVE_DF_CLUSTER, SAVE_DF_LOCALLY, BATCH_SIZE_LOCALLY, BATCH_SIZE_CLUSTER, FOLDER_TO_SAVE_CLUSTER, FOLDER_TO_SAVE_LOCALLY, NUM_LAYERS_CLUSTER, NUM_LAYERS_LOCALLY
from Curation import Study
import os
import random
import re
from loguru import logger
import gc 
import datetime

# STATE_RUN = 'locally'
STATE_RUN = 'cs_computer'


def categorize_numbers(numbers, sorted_indices) -> Dict[int, List[int]]:
    categorized = {key: [] for key in sorted_indices}
    for number in numbers:
        for index in sorted_indices:
            if number <= index:
                categorized[index].append(number)
                break
    return categorized


def get_numbers_files(path_to_embeddings) -> Dict[int, str]:
    filenames = os.listdir(path_to_embeddings)
    index_to_filename = {int(re.findall(r'\d+', name)[0]): name for name in filenames if name[-3:] == 'csv'}
    return index_to_filename


def get_random_indexes(common_elements, df_uncertain, df_usable) -> List[int]:
    # df_uncertain_filtered = df_uncertain[~df_uncertain['AASeq'].isin(common_elements['AASeq'])]
    # df_usable_filtered = df_usable[~df_usable['AASeq'].isin(common_elements['AASeq'])]
    amount_random_numbers = len(df_usable) - len(common_elements) + (((len(common_elements) / len(df_uncertain))) * 100)
    amount_random_numbers = amount_random_numbers * len(NUM_LAYERS_CLUSTER)
    random_numbers = [random.randint(0, len(df_uncertain) - 1) for _ in range(int(amount_random_numbers))]
    sorted_random_numbers = sorted(random_numbers)
    return sorted_random_numbers


def get_path_to_embddings() -> str:
    if STATE_RUN == 'locally':
        path_to_embeddings = FOLDER_TO_SAVE_LOCALLY
    elif STATE_RUN == 'cs_computer':
        path_to_embeddings = FOLDER_TO_SAVE_CLUSTER
    else:
        raise(f"state run is {STATE_RUN} and its not locally nor cs_computer")
    return path_to_embeddings


def save_df(df, path_to_embeddings):
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    df.to_csv(os.path.join(path_to_embeddings, "data_for_tsne", f"data_{formatted_time}.csv"))



def get_df(categorized_numbers, index_to_filename, path_to_embeddings):
    df = pd.DataFrame()
    for index, numbers in categorized_numbers.items():

        if numbers:  # Only process if there are numbers for this file
            filename = index_to_filename[index]
            logger.info(f"Processing file {filename} for indices {numbers}")
            curr_df = pd.read_csv(os.path.join(path_to_embeddings, filename))
            if STATE_RUN == 'locally':
                start_index = index - SAVE_DF_LOCALLY - BATCH_SIZE_LOCALLY
            else:
                start_index = index - SAVE_DF_CLUSTER - BATCH_SIZE_CLUSTER
            curr_df['change'] = (curr_df['seq'] != curr_df['seq'].shift(1)).astype(int)
            curr_df['cumulative'] = curr_df['change'].cumsum()
            curr_df['indexes'] = curr_df['cumulative'] + start_index - 1
            filtered_df = curr_df[curr_df['indexes'].isin(numbers)]
            filtered_df = filtered_df.drop(['change', 'cumulative', 'indexes'], axis=1)
            if df.empty:
                df = filtered_df
            else:
                df = pd.concat([df, filtered_df], ignore_index=True)

            del curr_df
            del filtered_df
            gc.collect()

    return df


def main():
    study_uncertain = Study(STUDY_ID)
    df_uncertain = study_uncertain.build_train_representations(samples=study_uncertain._samples['uncertain'], save=False, path=None)
    study_usable = Study(STUDY_ID)
    df_usable = study_usable.build_train_representations(samples=None, save=False, path=None)

    common_elements = pd.merge(df_uncertain, df_usable, on='AASeq')

    sorted_random_numbers = get_random_indexes(common_elements, df_uncertain, df_usable)
    path_to_embeddings = get_path_to_embddings()
    index_to_filename = get_numbers_files(path_to_embeddings)
    sorted_indices = sorted(index_to_filename.keys())
    categorized_numbers = categorize_numbers(sorted_random_numbers, sorted_indices)
    
    df = get_df(categorized_numbers, index_to_filename, path_to_embeddings)
    save_df(df, path_to_embeddings)


if __name__ == '__main__':
    current_time = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logger.add(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/loggers/create_data_from_blood/logfile_{STATE_RUN}_{time_string}.log", format="{time} {level} {message}")

    main()