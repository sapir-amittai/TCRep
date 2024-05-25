from Curation import Study

STUDY_ID = 'PRJNA393498'

if __name__ == '__main__':
    study = Study(STUDY_ID)
    # return numpy array of unique sequences - see docstring
    sequence = study.build_train_representations(samples=None, save=False, path=None)
    print(sequence)
    # use esm_2 to extract representation
    model_checkpoint = "facebook/esm2_t4815B__UR50D"
    # pad to length of the longest sequence - can be done through esm parameters (no need to change sequences)
    # save a dictionary of {seq : representation} - make sure to save on lab folder may be heavy

