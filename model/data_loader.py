import torch
from torch.utils.data import Dataset, IterableDataset
import os 
from model_cfg import CFG

class DeePEF(Dataset):
    '''
    Deep energy function dataset.
    Data item structure:
        # 3D coordinates of the protein
            * coordsAlpha
            * coordsBeta 
            * coordsC
            * coordsCa
            * coordsN
            * coordsAlpha native
            * coordsBeta native
            * coordsC native
            * coordsCa native
            * coordsN native
        # Embeddings
           * Large languege model embeddings
           * hand selected features
    '''
    def __init__(self, datapath=CFG.data_path,homothresh=CFG.homothresh,type='train'):
        """_summary_
        Data set for the Deep energy function dataset.
        Args:
            datapath (string,): _description_. Defaults to CFG.data_path.
            homothresh (float, optional): _description_. Defaults to CFG.homothresh.
            type (str, optional): _description_. Defaults to 'train'.
        """
        self.datapath = datapath
        self.homothresh = homothresh
        self.type = type

    def __len__(self):
        return len(os.listdir(self.datapath))

    def __getitem__(self, index):
        
        index_path = os.path.join(self.datapath, str(index))
       # Sequence of the protein
        seq = torch.load(os.path.join(index_path, 'seq.pt')).to(CFG.device)
        id = torch.load(os.path.join(index_path, 'ids.pt')).to(CFG.device)
       # 3D coordinates of the protein
        coordsAlpha = torch.load(os.path.join(index_path, 'coordsAlpha.pt')).to(CFG.device)
        coordsBeta = torch.load(s.path.join(index_path, 'coordsBeta.pt')).to(CFG.device)
        coordsC = torch.load(os.path.join(index_path, 'coordsC.pt')).to(CFG.device)
        coordsCa = torch.load(os.path.join(index_path, 'coordsCa.pt')).to(CFG.device)
        coordsN = torch.load(os.path.join(index_path, 'coordsN.pt')).to(CFG.device)
        # 3D coordinates of the protein native
        coordsAlpha_native = torch.load(os.path.join(index_path, 'coordsAlpha_native.pt')).to(CFG.device)
        coordsBeta_native = torch.load(os.path.join(index_path, 'coordsBeta_native.pt')).to(CFG.device)
        coordsC_native = torch.load(os.path.join(index_path, 'coordsC_native.pt')).to(CFG.device)
        coordsCa_native = torch.load(os.path.join(index_path, 'coordsCa_native.pt')).to(CFG.device)
        coordsN_native = torch.load(os.path.join(index_path, 'coordsN_native.pt')).to(CFG.device)
        # Masks
        mask = torch.load(os.path.join(index_path, 'mask.pt')).to(CFG.device)
        nativemask = torch.load(os.path.join(index_path, 'nativemask.pt')).to(CFG.device)
        # Embeddings
        esm_embed = torch.load(os.path.join(index_path, 'emb_esm.pt')).to(CFG.device)
        
        return seq, id, coordsAlpha, coordsBeta, coordsC, coordsCa, coordsN, coordsAlpha_native, coordsBeta_native, coordsC_native, coordsCa_native, coordsN_native, mask, nativemask, esm_embed 
        