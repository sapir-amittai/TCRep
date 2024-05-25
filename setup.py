from definitions import *
import wget
from Curation import build_study


def create_directories():
    os.mkdir(STUDIES_DATABASE)
    os.chdir(pjoin(BASE_DIRECTORY, STUDIES_DATABASE))
    for db in TCR_DATABASES.values():
        os.mkdir(db)
        with open(INDEX, 'w') as _:
            pass
    os.chdir(BASE_DIRECTORY)
    os.mkdir(OBJECTS_DATABASE)
    os.chdir(pjoin(BASE_DIRECTORY, STUDIES_DATABASE))
    for db in OBJECTS_TYPES:
        os.mkdir(db)
    os.chdir(BASE_DIRECTORY)


def download_studies():
    for study_id in INIT_STUDIES:
        url = TCRDB_DOWNLOAD_URL + study_id
        wget.download(url, out=TCR_DB_PATH)


if __name__ == '__main__':
    create_directories()
    download_studies()
    #  build AS study
    build_study('PRJNA393498',
                "This study was aimed to search for AS-specific T cell receptor (TCR) variants, to determine the phenotype and involvement of corresponding T-cells in joint inflammation",
                ['AASeq', 'Vregion', 'Dregion', 'Jregion', 'RunId'],
                ['SRR5812617', 'SRR5812618', 'SRR5812627', 'SRR5812653', 'SRR5812656', 'SRR5812663', 'SRR5812665', 'SRR5812666', 'SRR5812676'],
                ['SRR5812612', 'SRR5812623', 'SRR5812669', 'SRR5812671', 'SRR5812637', 'SRR5812668', 'SRR5812657', 'SRR5812640', 'SRR5812672', 'SRR5812616', 'SRR5812687', 'SRR5812648', 'SRR5812651', 'SRR5812677', 'SRR5812626', 'SRR5812643', 'SRR5812624', 'SRR5812644', 'SRR5812645', 'SRR5812678', 'SRR5812655', 'SRR5812683', 'SRR5812682', 'SRR5812613', 'SRR5812667', 'SRR5812625', 'SRR5812649', 'SRR5812674', 'SRR5812611', 'SRR5812610', 'SRR5812684', 'SRR5812622', 'SRR5812654', 'SRR5812658', 'SRR5812686', 'SRR5812662', 'SRR5812636', 'SRR5812660', 'SRR5812633', 'SRR5812679', 'SRR5812634', 'SRR5812646', 'SRR5812635', 'SRR5812620', 'SRR5812681', 'SRR5812652', 'SRR5812685', 'SRR5812675', 'SRR5812614', 'SRR5812680', 'SRR5812642', 'SRR5812621', 'SRR5812630', 'SRR5812650', 'SRR5812664', 'SRR5812639', 'SRR5812670', 'SRR5812659', 'SRR5812638', 'SRR5812629', 'SRR5812641', 'SRR5812661', 'SRR5812673', 'SRR5812631', 'SRR5812647', 'SRR5812619', 'SRR5812632', 'SRR5812628', 'SRR5812615'],
                [])
