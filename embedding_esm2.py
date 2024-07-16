# taken from https://github.com/facebookresearch/esm?tab=readme-ov-file#bulk_fasta
from typing import List
import torch
import esm
from loguru import logger



def run_model(model, alphabet, num_layers: List[int], sequence_data: List[str], mean_embedding: bool):
    num_layers.sort()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # We don't need to use <mask> tokens (as shown in the github example)
    data = [(f"protein{i}", seq) for i, seq in enumerate(sequence_data)]

    #  representation will automatically be padded to length of the longest sequence in the batch
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    if torch.cuda.is_available():
        model = model.cuda()
        batch_tokens = batch_tokens.cuda()  # Move tokens to GPU
        logger.info("Transferred model and tokens to GPU")
    else:
        logger.warning("running on CPU")
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=num_layers, return_contacts=True)

    #  representation are taken from last layer - we may want to change it
    #  try to also save representations from layer 27 - think this can be done with repr_layers=[33,27]
    #  then also token_representations_27 = results["representations"][27]
    
    sequence_representations = {}
    for num_layer in num_layers:
        token_representations = results["representations"][num_layer]

        # Generate per-sequence representations via averaging
        # This is the easiest way to deal with sequences of unequal length (just take the average)
        # we may want to find a more sophisticated way to deal with it in the future
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations_tmp = []
        for j, tokens_len in enumerate(batch_lens):
            if mean_embedding:
                new_seq = token_representations[j, 1 : tokens_len - 1].mean(dim=0, keepdim=True)
            else:
                new_seq = token_representations[j, 1 : tokens_len - 1]
            sequence_representations_tmp.append(new_seq)
        sequence_representations[num_layer] = sequence_representations_tmp

    return sequence_representations



if __name__ == '__main__':
    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    sequence_data = ["CATSDLDSSYNEQFF", "CASSSGVRVSGANVLTF"]
    run_model(model, alphabet, [4, 5, 6], sequence_data, mean_embedding=False)