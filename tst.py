import torch
from tqdm import tqdm
import numpy as np
import os

def main():
# Move emmbedings to data2:
# '''copy emmbeding from data to data2'''
    emb_path = "./data/esm2_t12_35M_UR50D/"
    emb_new_path = "./data/data2/"
    file_names = os.listdir(emb_new_path)
    for i in tqdm(np.arange(0,len(file_names),1)):
        file = torch.load(emb_path+"emb_"+str(i)+".pt")
        torch.save(file,emb_new_path+str(i)+'/'+"emb_esm.pt")

if __name__ == '__main__':
    main()