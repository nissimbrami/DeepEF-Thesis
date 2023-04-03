import torch

class CFG:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable
    if(device.type == "cuda" or device.type == "mps"):
        torch.cuda.empty_cache()
        cuda = True
    else:
        cuda = False
        
    data_path = './data/data2'
    seed = 42
    debug = False
    # Train data parameters
    homothresh = 0.9
    constraint = True
    split_train = 0.8
    debuge_size  = 100
    # Model parameters
    h = 0.001
    coords_emb = 64
    embedding_size = 20
    filters = 64
    num_layers = 1
    model_path = "./trianed_models_no_const/"
    gaussian_coef = -0.008*1e1
    #training parameters
    lr = 0.0001
    wd = 0.0001
    batch_size = 1
    num_workers = 2
    N = 10
    num_epochs = 5