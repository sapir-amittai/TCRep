from Curation import Study

STUDY_ID = 'PRJNA393498'

if __name__ == '__main__':
    study = Study(STUDY_ID)
    # use default parameters see docstring - return numpy array of unique sequences
    sequence = study.build_train_representations(samples=None, save=bool, path=None)

    # use esm_2 to extract representation
    # pad to length of the longest sequence - can be done through esm parameters (no need to change sequences)
    # save a dictionary of {seq : representation} - make sure to save on lab folder may be heavy

