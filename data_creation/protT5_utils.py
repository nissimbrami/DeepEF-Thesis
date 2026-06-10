from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


def load_model(model="Rostlab/prot_t5_xl_half_uniref50-enc"):
    transformer_link = model
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)
    # model.full() if device == 'cpu' else model.half()  # only cast to full-precision if no GPU is available
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    print(f'loaded model {model}')
    return model, tokenizer


def get_embeddings(model, tokenizer, sequences):
    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    return embedding_repr.last_hidden_state


if __name__ == '__main__':
    sequence_examples = ['LKVIWFIHVIKLE', 'CCMPRWCWPKLPP']
    model, tokenizer = load_model()
    embeddings = get_embeddings(model, tokenizer, sequence_examples)
    print(embeddings.shape)
