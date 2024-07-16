import pandas as pd
from consts import STUDY_ID
from Curation import Study
import os
from utils import get_filtered_and_mean_file_bood, get_filtered_and_mean_file_fluid
from consts import PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD, PATH_TO_CSV_FLUID


def final_data_to_tsne_blood(common_elements):
    df_blood = pd.read_csv(os.path.join(PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD))
    df_filtered_common_blood = df_blood[~df_blood['seq'].isin(common_elements['AASeq'])]
    average_embeddings = df_filtered_common_blood.groupby(['seq', 'esm2_layer']).mean().reset_index()
    average_embeddings.to_csv(get_filtered_and_mean_file_bood(PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD))


def final_data_to_tsne_fluid(common_elements):
    filenames = os.listdir(PATH_TO_CSV_FLUID)

    df_fluid = pd.DataFrame()
    for filename in filenames:
        filename_csv_blood = os.path.join(PATH_TO_CSV_FLUID, filename)
        sub_df_fluid = pd.read_csv(os.path.join(PATH_TO_CSV_FLUID, filename_csv_blood))
        sub_df_filtered_common_fluid = sub_df_fluid[~sub_df_fluid['seq'].isin(common_elements['AASeq'])]
        sub_average_embeddings = sub_df_filtered_common_fluid.groupby(['seq', 'esm2_layer']).mean().reset_index()
        if df_fluid.empty:
            df_fluid = sub_average_embeddings
        else:
            df_fluid = pd.concat([df_fluid, sub_average_embeddings], ignore_index=True)
    df_fluid.to_csv(get_filtered_and_mean_file_fluid(PATH_TO_CSV_FLUID))


def main():
    study_uncertain = Study(STUDY_ID)
    df_uncertain = study_uncertain.build_train_representations(samples=study_uncertain._samples['uncertain'], save=False, path=None)
    study_usable = Study(STUDY_ID)
    df_usable = study_usable.build_train_representations(samples=None, save=False, path=None)
    common_elements = pd.merge(df_uncertain, df_usable, on='AASeq')

    # final_data_to_tsne_fluid(common_elements)
    final_data_to_tsne_blood(common_elements)
    




if __name__ == '__main__':
    # current_time = time.localtime()  # get struct_time
    # time_string = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    # logger.add(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/loggers/create_data_from_blood/logfile_{STATE_RUN}_{time_string}.log", format="{time} {level} {message}")

    main()