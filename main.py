from Curation import Study
from trainer import get_df_with_esm_results
import esm
from loguru import logger
import time
from consts import (
    SAVE_DF_LOCALLY, BATCH_SIZE_LOCALLY, NUM_LAYERS_LOCALLY, FOLDER_TO_SAVE_LOCALLY,
    SAVE_DF_CLUSTER, BATCH_SIZE_CLUSTER, NUM_LAYERS_CLUSTER, FOLDER_TO_SAVE_CLUSTER,
    STUDY_ID, MEAN_EMBEDDING_LOCALLY, MEAN_EMBEDDING_CLUSTER
)


def run_locally(sequence, prefix_filepath_save=''):
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    sequences = sequence['AASeq'].to_list()
    start_batch = 0
    get_df_with_esm_results(
        model, alphabet, sequences, num_layers=NUM_LAYERS_LOCALLY, batch_size=BATCH_SIZE_LOCALLY, 
        folder_to_save=FOLDER_TO_SAVE_LOCALLY, when_save_df=SAVE_DF_LOCALLY, 
        start_batch=start_batch, mean_embedding=MEAN_EMBEDDING_LOCALLY, prefix_filepath_save=prefix_filepath_save
        )


def run_on_cs_computers(sequence, prefix_filepath_save=''):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    sequences = sequence['AASeq'].to_list()
    start_batch = 0
    get_df_with_esm_results(
        model, alphabet, sequences, num_layers=NUM_LAYERS_CLUSTER, batch_size=BATCH_SIZE_CLUSTER, 
        folder_to_save=FOLDER_TO_SAVE_CLUSTER, when_save_df=SAVE_DF_CLUSTER, break_after_save=False, 
        start_batch=start_batch, mean_embedding=MEAN_EMBEDDING_CLUSTER, prefix_filepath_save=prefix_filepath_save
        )



if __name__ == '__main__':
    current_time = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logger.add(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/loggers/run_esm2/logfile_{time_string}.log", format="{time} {level} {message}")

    study = Study(STUDY_ID)
    # return numpy array of unique sequences - see docstring
    # sequence = study.build_train_representations(samples=study._samples['uncertain'], save=False, path=None)
    # sequence = study.build_train_representations(save=False, path=None)
    # sequence = study.get_df_same_type_same_t_cell(type='SF', sub_family_t='4')
    sequence_sf_4 = study.get_repeat_sequence(type='SF', sub_family_t='4')
    sequence_sf_8 = study.get_repeat_sequence(type='SF', sub_family_t='8')
    sequence_pb_4 = study.get_repeat_sequence(type='PB', sub_family_t='4')
    sequence_pb_8 = study.get_repeat_sequence(type='PB', sub_family_t='8')

    
    # run_locally(sequence_sf_4.head(20), 'SF_4_')
    run_on_cs_computers(sequence_sf_4, 'SF_4_')
    run_on_cs_computers(sequence_sf_8, 'SF_8_')
    run_on_cs_computers(sequence_pb_4, 'PB_4_')
    run_on_cs_computers(sequence_pb_8, 'PB_8_')




    # print(sequence)
    # sequence.to_csv("save_seq.csv")
    # use esm_2 to extract representation
    # model_checkpoint = "facebook/esm2_t4815B__UR50D"
    # pad to length of the longest sequence - can be done through esm parameters (no need to change sequences)
    # save a dictionary of {seq : representation} - make sure to save on lab folder may be heavy

