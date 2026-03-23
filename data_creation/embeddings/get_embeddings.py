import torch
import esm
import numpy as np
import gc
from tqdm import tqdm
import logging
import os.path

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable

# Create logger
logger = logging.getLogger()

def logger_setup():   
    fhandler = logging.FileHandler(filename='mylog.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    logger.debug("started run")


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
    model_name = "esm2_t12_35M_UR50D"#"esm2_t30_150M_UR50D"
    num_layers = 12
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
    # all_data = all_data[1748:]
    start_index = 1#1748
    batch_size = 1
    for item_i in tqdm(np.arange(start_index,len(all_data),batch_size)):
        #print_gpu()
        torch.cuda.empty_cache()
        gc.collect()
        data = all_data[item_i-batch_size:item_i]
        if not os.path.isfile(f"./data/{model_name}/emb_{item_i}.pt"): 
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)# move to GPU
            # Extract per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[num_layers], return_contacts=True)
            
            token_representations = results["representations"][num_layers].to(device="cpu")
                    
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            # seq_emb.extend(sequence_representations)
            torch.save(sequence_representations,f"./data/{model_name}/emb_{item_i}.pt")
            del sequence_representations,token_representations,batch_lens,batch_labels, batch_strs, batch_tokens,results
            torch.cuda.empty_cache()
            gc.collect()
def main():
    create_emmbeding(log_file=True,debuge=False)

if __name__=='__main__':
    main()
