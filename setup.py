from definitions import *
import wget
from Curation import build_study


def create_directories():
    print("Creating directories...")
    os.mkdir(STUDIES_DATABASE)
    os.chdir(pjoin(BASE_DIRECTORY, STUDIES_DATABASE))
    for db in TCR_DATABASES.values():
        os.mkdir(db)
        with open(pjoin(db, INDEX), 'w') as _:
            pass
    os.chdir(BASE_DIRECTORY)
    os.mkdir(OBJECTS_DATABASE)
    os.chdir(pjoin(BASE_DIRECTORY, OBJECTS_DATABASE))
    for db in OBJECTS_TYPES:
        os.mkdir(db)
    os.chdir(BASE_DIRECTORY)
    print('done')


def download_studies():
    print('downloading studies...')
    for study_id in INIT_STUDIES:
        url = TCRDB_DOWNLOAD_URL + study_id + '.tsv'
        print(f"{url}")
        wget.download(url, out=TCR_DB_PATH)
    print('done')


if __name__ == '__main__':
    create_directories()
    download_studies()
    #  build AS study

    #  build AS study
    build_study('PRJNA393498',
                "This study was aimed to search for AS-specific T cell receptor (TCR) variants, to determine the phenotype and involvement of corresponding T-cells in joint inflammation",
                ['AASeq', 'Vregion', 'Dregion', 'Jregion', 'RunId'],
                [['SRR5812617', 'Synovial fluid'], ['SRR5812618','Synovial fluid'], ['SRR5812627', 'Synovial fluid'],
                 ['SRR5812653', 'Synovial fluid'], ['SRR5812656', 'Synovial fluid'], ['SRR5812663', 'Synovial fluid'],
                 ['SRR5812665', 'Synovial fluid'], ['SRR5812666', 'Synovial fluid'], ['SRR5812676', 'Synovial fluid']],
                [['SRR5812612', 'blood'], ['SRR5812623', 'blood'], ['SRR5812669', 'blood'], ['SRR5812671', 'blood'],
                 ['SRR5812637', 'blood'], ['SRR5812668', 'blood'], ['SRR5812657', 'blood'], ['SRR5812640', 'blood'],
                 ['SRR5812672', 'blood'], ['SRR5812616', 'blood'], ['SRR5812687', 'blood'], ['SRR5812648', 'blood'],
                 ['SRR5812651', 'blood'], ['SRR5812677', 'blood'], ['SRR5812626', 'blood'], ['SRR5812643', 'blood'],
                 ['SRR5812624', 'blood'], ['SRR5812644', 'blood'], ['SRR5812645', 'blood'], ['SRR5812678', 'blood'],
                 ['SRR5812655', 'blood'], ['SRR5812683', 'blood'], ['SRR5812682', 'blood'], ['SRR5812613', 'blood'],
                 ['SRR5812667', 'blood'], ['SRR5812625', 'blood'], ['SRR5812649', 'blood'], ['SRR5812674', 'blood'],
                 ['SRR5812611', 'blood'], ['SRR5812610', 'blood'], ['SRR5812684', 'blood'], ['SRR5812622', 'blood'],
                 ['SRR5812654', 'blood'], ['SRR5812658', 'blood'], ['SRR5812686', 'blood'], ['SRR5812662', 'blood'],
                 ['SRR5812636', 'blood'], ['SRR5812660', 'blood'], ['SRR5812633', 'blood'], ['SRR5812679', 'blood'],
                 ['SRR5812634', 'blood'], ['SRR5812646', 'blood'], ['SRR5812635', 'blood'], ['SRR5812620', 'blood'],
                 ['SRR5812681', 'blood'], ['SRR5812652', 'blood'], ['SRR5812685', 'blood'], ['SRR5812675', 'blood'],
                 ['SRR5812614', 'blood'], ['SRR5812680', 'blood'], ['SRR5812642', 'blood'], ['SRR5812621', 'blood'],
                 ['SRR5812630', 'blood'], ['SRR5812650', 'blood'], ['SRR5812664', 'blood'], ['SRR5812639', 'blood'],
                 ['SRR5812670', 'blood'], ['SRR5812659', 'blood'], ['SRR5812638', 'blood'], ['SRR5812629', 'blood'],
                 ['SRR5812641', 'blood'], ['SRR5812661', 'blood'], ['SRR5812673', 'blood'], ['SRR5812631', 'blood'],
                 ['SRR5812647', 'blood'], ['SRR5812619', 'blood'], ['SRR5812632', 'blood'], ['SRR5812628', 'blood'],
                 ['SRR5812615', 'blood']],
                [])