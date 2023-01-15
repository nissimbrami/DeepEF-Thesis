import torch
from torch.utils.data import Dataset, DataLoader
import os 
from model_cfg import CFG
from sklearn.model_selection import train_test_split

class PEFDataset(Dataset):
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
    def __init__(self,file_df ,datapath=CFG.data_path,homothresh=CFG.homothresh,type='train'):
        """_summary_
        Data set for the Deep energy function dataset.
        Args:
            file_df (list): _description_.files dataframe from the data path
            datapath (string,): _description_. Defaults to CFG.data_path.
            homothresh (float, optional): _description_. Defaults to CFG.homothresh.
            type (str, optional): _description_. Defaults to 'train'.
        """
        self.datapath = datapath
        self.filenames = file_df
        self.homothresh = homothresh
        self.type = type

    def __len__(self):
        return len(os.listdir(self.datapath))

    def __getitem__(self, index):
        
        index_path = os.path.join(self.datapath, self.filenames[index])
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
        
def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    # Get the filenames from the train folder
    file_names = os.listdir(data_dir)
    # Split the data into train, validation and test set
    X_train, X_rem, y_train, y_rem = train_test_split(file_names,file_names, train_size=CFG.split_train,
                                                      random_state=CFG.seed)
    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
    # Now we have the data split in training, validation and test set
    dataloaders['train']= DataLoader(PEFDataset(X_train,datapath=data_dir), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    dataloaders['val']= DataLoader(PEFDataset(X_valid,datapath=data_dir), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)

    dataloaders['test']= DataLoader(PEFDataset(X_test,datapath=data_dir), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    return dataloaders

#TODO: clean dataset from  homology threshold