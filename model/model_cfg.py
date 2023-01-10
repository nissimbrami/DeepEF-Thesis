import torch

class CFG:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable
    data_path = './data/data2'
    # Train data parameters
    homothresh = 0.9
    
    # Model parameters