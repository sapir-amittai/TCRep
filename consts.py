import pandas as pd


# PATH_TO_CSV_BLOOD = "/cs/labs/dina/sapir_amittai/code/spring_24/esm_2_embedding_blood/run_on_cs_computer/data_for_tsne"
# # FILENAME_CSV_BLOOD = "data_20240625_173958.csv"
# FILENAME_CSV_BLOOD = "data_2024-06-30_20:22:14.csv"
# PATH_TO_CSV_FLUID = "/cs/labs/dina/sapir_amittai/code/spring_24/esm_2_embedding_synovial_fluid /run_on_cs_computer"
# HOW_TO_COMBINE_EMBEDDINGS = "filtered_and_mean"

PATH_TO_CSV_BLOOD = "/cs/labs/dina/sapir_amittai/code/spring_24/esm_2_embedding_blood/run_on_cs_computer/embedding_duplicate_seq_same_type"
FILENAME_CSV_BLOOD = "PB_4_until_seq_1500.csv"
PATH_TO_CSV_FLUID = "/cs/labs/dina/sapir_amittai/code/spring_24/esm_2_embedding_blood/run_on_cs_computer/embedding_duplicate_seq_same_type"
HOW_TO_COMBINE_EMBEDDINGS = "SF_4_until_seq_500.csv"

STUDY_ID = 'PRJNA393498'
# locally
SAVE_DF_LOCALLY = None # 6
BATCH_SIZE_LOCALLY = 2
NUM_LAYERS_LOCALLY = [4, 6]
FOLDER_TO_SAVE_LOCALLY = "/cs/labs/dina/sapir_amittai/code/spring_24/esm_2_embedding_blood/run_locally"
MEAN_EMBEDDING_LOCALLY = True

# cluster
SAVE_DF_CLUSTER = None # 10_000
BATCH_SIZE_CLUSTER = 500
NUM_LAYERS_CLUSTER = [27, 33]
FOLDER_TO_SAVE_CLUSTER = "/cs/labs/dina/sapir_amittai/code/spring_24/esm_2_embedding_blood/run_on_cs_computer/embedding_duplicate_seq_same_type"
MEAN_EMBEDDING_CLUSTER = True

# data - synovial fluid
COLUMNS_DATA_ID = ['RunId', 'name', 'type', 'SubFamilyT']  # sub family t is sub family in T cells
synovial_fluid_id = [
    ['SRR5812676', 'Dv', 'SF', '4'],
    ['SRR5812666', 'Kal', 'SF', '8'],
    ['SRR5812665', 'Kal', 'SF', '4'],
    # ['SRR5812663', 'Kos', 'SF', 'TRBV9'],
    ['SRR5812656', 'Mikh', 'SF', '4'],
    ['SRR5812653', 'Mikh', 'SF', '8'],
    ['SRR5812627', 'Dv', 'SF', '8'],
    ['SRR5812618', 'Shep', 'SF', '4'],
    ['SRR5812617', 'Shep', 'SF', '8']
]
SYNOVIAL_FLUID_ID = pd.DataFrame(synovial_fluid_id, columns=COLUMNS_DATA_ID) 
blood_id = [
    ['SRR5812620', 'Shep', 'PB', '8'],
    ['SRR5812619', 'Shep', 'PB', '4'],
    ['SRR5812678', 'Dv', 'PB', '4'],
    ['SRR5812675', 'Dv', 'PB', '8'],
    ['SRR5812654', 'Mikh', 'PB', '8'],
    ['SRR5812651', 'Mikh', 'PB', '4'],
    ['SRR5812668', 'Kal', 'PB', '4'],
    ['SRR5812667', 'Kal', 'PB', '8']
]
BLOOD_ID = pd.DataFrame(blood_id, columns=COLUMNS_DATA_ID)
DATA_ID = pd.concat([SYNOVIAL_FLUID_ID, BLOOD_ID], ignore_index=True)