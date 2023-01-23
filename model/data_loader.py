import torch
from torch.utils.data import Dataset, DataLoader
import os 
from model.model_cfg import CFG
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    def __init__(self,file_df ,datapath=CFG.data_path,homothresh=CFG.homothresh,type='train',train_type = 'AlphaFold'):
        """_summary_
        Data set for the Deep energy function dataset.
        Args:
            file_df (list): _description_.files dataframe from the data path
            datapath (string,): _description_. Defaults to CFG.data_path.
            homothresh (float, optional): _description_. Defaults to CFG.homothresh.
            type (str, optional): _description_. Defaults to 'train'.
        """
        self.datapath = datapath
        self.filenames = file_df.copy()
        self.homothresh = homothresh
        self.type = type
        self.train_type = train_type
        # remove the files with homology greater than homothresh
        if type == 'train':
            self.check_data_constrain()
        

    def __len__(self):
        return len(os.listdir(self.datapath))

    def __getitem__(self, index):
        
        index_path = os.path.join(self.datapath, self.filenames[index])
       # Sequence of the protein
        seq = torch.load(os.path.join(index_path, 'seq.pt')).to(CFG.device)
        id = torch.load(os.path.join(index_path, 'ids.pt')) # string id
       # 3D coordinates of the protein
        coordsAlpha = torch.load(os.path.join(index_path, 'CoordAlpha.pt')).to(CFG.device)
        coordsBeta = torch.load(os.path.join(index_path, 'CoordBeta.pt')).to(CFG.device)
        coordsC = torch.load(os.path.join(index_path, 'CoordC.pt')).to(CFG.device)
        coordsN = torch.load(os.path.join(index_path, 'CoordN.pt')).to(CFG.device)
        # 3D coordinates of the protein native
        coordsAlpha_native = torch.load(os.path.join(index_path, 'CoordCaNative.pt')).to(CFG.device)
        coordsBeta_native = torch.load(os.path.join(index_path, 'CoordCbNative.pt')).to(CFG.device)
        coordsC_native = torch.load(os.path.join(index_path, 'CoordCNative.pt')).to(CFG.device)
        coordsN_native = torch.load(os.path.join(index_path, 'CoordNNative.pt')).to(CFG.device)
        # Masks
        mask = torch.load(os.path.join(index_path, 'mask.pt')).to(CFG.device)
        nativemask = torch.load(os.path.join(index_path, 'nativemask.pt')).to(CFG.device)
        # Embeddings
        esm_embed = torch.load(os.path.join(index_path, 'emb_esm.pt'))[0].to(CFG.device)
        # Concatenate the coordinates
        Xd = self.concat_cords(coordsAlpha,coordsBeta, coordsC, coordsN)
        Xn = self.concat_cords(coordsAlpha_native,coordsBeta_native, coordsC_native, coordsN_native)
            
        return seq, id, Xd,Xn, mask, nativemask, esm_embed 
        
    def read_protein(self,index):
        """
        Read the protein data from the index path

        Args:
            index (int): index of protein in the dataset
        """
        index_path = os.path.join(self.datapath, self.filenames[index])
        # Sequence of the protein
        seq = torch.load(os.path.join(index_path, 'seq.pt')).to(CFG.device)
        id = torch.load(os.path.join(index_path, 'ids.pt')) # string id
        # 3D coordinates of the protein
        coordsAlpha = torch.load(os.path.join(index_path, 'CoordAlpha.pt')).to(CFG.device)
        coordsBeta = torch.load(os.path.join(index_path, 'CoordBeta.pt')).to(CFG.device)
        coordsC = torch.load(os.path.join(index_path, 'CoordC.pt')).to(CFG.device)
        coordsN = torch.load(os.path.join(index_path, 'CoordN.pt')).to(CFG.device)
        # 3D coordinates of the protein native
        coordsAlpha_native = torch.load(os.path.join(index_path, 'CoordCaNative.pt')).to(CFG.device)
        coordsBeta_native = torch.load(os.path.join(index_path, 'CoordCbNative.pt')).to(CFG.device)
        coordsC_native = torch.load(os.path.join(index_path, 'CoordCNative.pt')).to(CFG.device)
        coordsN_native = torch.load(os.path.join(index_path, 'CoordNNative.pt')).to(CFG.device)
        # Masks
        mask = torch.load(os.path.join(index_path, 'mask.pt')).to(CFG.device)
        nativemask = torch.load(os.path.join(index_path, 'nativemask.pt')).to(CFG.device)
        # Embeddings
        esm_embed = torch.load(os.path.join(index_path, 'emb_esm.pt'))[0].to(CFG.device)
        
        return seq, id, coordsAlpha,coordsBeta, coordsC, coordsN, coordsAlpha_native,coordsBeta_native, coordsC_native, coordsN_native, mask, nativemask, esm_embed
    
    def concat_cords(self,coordsAlpha,coordsBeta, coordsC, coordsN):
        """
        Concatenate the coordinates
        output: X (torch.tensor): concatenated coordinates [N,4,3]
        """
        coordsAlpha,coordsBeta, coordsC, coordsN = coordsAlpha.unsqueeze(1), coordsBeta.unsqueeze(1), coordsC.unsqueeze(1), coordsN.unsqueeze(1)
        X = torch.cat((coordsAlpha,coordsBeta, coordsC, coordsN), dim=1)
        return X
    
    def check_data_constrain(self):
        """
        Check the data constrain and remove the files with homology greater than homothresh.
        Check mask and native mask.
        Update file names list.
        """
        print('Checking data constrain...')
        new_filenames = []
        for i in tqdm(range(len(self.filenames))):
            (seq, id, coordAlpha,coordBeta, coordC, coordN, coordAlphaNative,
             coordBetaNative, coordCNative, coordNNative, mask, nativemask, embedding)  = self.read_protein(i)
            dt = torch.get_default_dtype()
            coordN = coordN.to(dt)
            coordAlpha = coordAlpha.to(dt)
            coordC = coordC.to(dt)
            coordBeta = coordBeta.to(dt)
            seq = seq.to(dt)
            embedding = embedding.to(dt)
            coordNNative = coordNNative.to(dt)
            coordAlphaNative = coordAlphaNative.to(dt)
            coordCNative = coordCNative.to(dt)
            coordBetaNative = coordBetaNative.to(dt)

            s = seq.mean(-1)
            if (self.homothresh is not None) and (s.max() > self.homothresh):
                # print("protein is too homogenuous", id)
                continue

            if (self.train_type is not None) and (not self.train_type in id):
                continue

            # scale = 1e-2
            # Mnat = nativemask
            # M = msk & Mnat

            # ind = torch.where(M)[0]
            # istart = ind[0]
            # ilast = ind[-1]
            # M = M[istart:ilast + 1]
            # msk = msk[istart:ilast + 1]
            # msk = msk.type('torch.FloatTensor')
            # if torch.any(msk == 0):
            #     # print("id problem", id)
            #     idx = (idx + 1) % self.__len__()
            #     continue
 
            new_filenames.append(self.filenames[i])
            
            torch.cuda.empty_cache()
            gc.collect()
        
        self.filenames = new_filenames
  
        
def fetch_dataloader(data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    # Get the filenames from the train folder
    file_names = os.listdir(data_dir)
    if params.debug:
        file_names = file_names[:100]
    # Split the data into train, validation and test set
    X_train, X_rem, y_train, y_rem = train_test_split(file_names,file_names, train_size=CFG.split_train,
                                                      random_state=CFG.seed)
    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
    # Now we have the data split in training, validation and test set
    train_loader= DataLoader(PEFDataset(X_train,datapath=data_dir), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    valid_loader= DataLoader(PEFDataset(X_valid,datapath=data_dir), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)

    test_loader= DataLoader(PEFDataset(X_test,datapath=data_dir), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    return train_loader, valid_loader, test_loader

#TODO: clean dataset from  homology threshold

class params:
    def __init__(self,batch_size,num_workers,cuda,debug=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda
        self.debug = debug
        
