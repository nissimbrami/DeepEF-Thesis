#@title Import dependencies. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
# import requests
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision) 
transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )

sequence_examples = ["PRTEINO", "GPSGLGLPAGLYAFNSGGISLDLGINDPVPFNTVGSQFGTAISQLDADTFVISETGFYKITVIANTATASVLGGLTIQVNGVPVPGTGSSLISLGAPIVIQAITQITTTPSLVEVIVTGLGLSLALGTSASIIIEKVA"]
def get_emb(sequence_examples):
    seq_len = len(sequence_examples[0])
    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

    # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7]) 
    emb_0 = embedding_repr.last_hidden_state[0,:seq_len]
    return emb_0

def print_files_in_directory(directory,mutation=False):
    n_saved = 0
    if mutation:
        emb_file_name = 'proT5_emb_mut.pt'
    else:
        emb_file_name = 'proT5_emb.pt'
    
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if (file_name == 'seq.pt'):
                seq = torch.load(os.path.join(root, file_name))
                if mutation:
                    mix_index = torch.randperm(len(seq))[:2] # randomly select 2 positions to swap
                    l1,l2 = seq[mix_index[0]], seq[mix_index[1]] # save the letters at these positions
                    # swap the letters at these positions
                    seq = seq[:mix_index[0]] + l2 + seq[mix_index[0]+1:]
                    seq = seq[:mix_index[1]] + l1 + seq[mix_index[1]+1:]
                proT5_emb = get_emb([seq]).to('cpu')
                torch.save(proT5_emb, os.path.join(root, emb_file_name))  
                n_saved += 1
                if (n_saved % 1000 == 0):
                    print('Saved {} files'.format(n_saved))
                

# Provide the directory path here
directory_path = './data/casp12_data_30/'

print_files_in_directory(directory_path,mutation=True)
