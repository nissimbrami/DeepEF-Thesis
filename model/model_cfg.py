import torch

class CFG:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable
    data_path = './data/data2'
    seed = 42
    # Train data parameters
    homothresh = 0.9
    split_train = 0.8
    # Model parameters
    h = 0.001