import torch
import esm
import numpy as np
import gc
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable

def print_gpu():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

def create_emmbeding(log_file=False,debuge=False):
    if log_file:
        logger_setup()
    logger.info(f"debuge mode {debuge}")
    # Load ESM-2 model
    primary_seq_path = "./data/seq_primar.txt"
    model_name = "esm2_t30_150M_UR50D"
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()  # disables dropout for deterministic results
    logger.info(f'Finished loading model - {model_name}')
    print('Finished loading model.')
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    all_data = []
    # Using readlines()
    file = open(primary_seq_path, 'r')
    index = 0
    while True:
        next_line = file.readline()
        if not next_line or (debuge and index==10):
            break 
        all_data.append([index,next_line])
        index+=1
    seq_emb = []
    batch_size = 1
    for i in tqdm(np.arange(batch_size,len(all_data),batch_size)):
        #print_gpu()
        torch.cuda.empty_cache()
        gc.collect()
        data = all_data[i-batch_size:i]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)# move to GPU
        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30], return_contacts=True)
        token_representations = results["representations"][30]
                
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        seq_emb.extend(sequence_representations)
        del sequence_representations,token_representations,batch_lens,batch_labels, batch_strs, batch_tokens,results
        torch.cuda.empty_cache()
        gc.collect()
        torch.save(seq_emb,"esm2_t30_150M_UR50D_emb2.pt")


def main():
    logger_setup()
    create_emmbeding(log_file=True,debuge=False)


def main():
    logger_setup()
    create_emmbeding(log_file=True,debuge=True)

if __name__=='__main__':
    main()
