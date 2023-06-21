import torch

class CFG:
    device =torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') #torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable
    if(device.type == "cuda" or device.type == "mps"):
        torch.cuda.empty_cache()
        cuda = True
    else:
        cuda = False
        
    data_path = './data/casp12_data_30/' #'./data/data2'
    inference_path = './data/inference_data'
    results_path = './res/results-emb/'
    seed = 42
    debug = True
    # Train data parameters
    homothresh = 0.9
    constraint = True
    split_train_size = 0.8
    debug_size  = 10
    sigma = 0.1 # for score matching loss
    # Model parameters
    h = 0.001
    coords_emb_size = 64
    embedding_size = 20
    filters = 64
    num_layers = 1
    model_path = "./res/trianed_models/"
    gaussian_coef = -0.008*1e1
    #training parameters
    lr = 0.0001
    wd = 0.0001
    batch_size = 1
    num_workers = 2
    N = 10
    num_epochs = 40
    seq_len = 1000
    SM = False # score matching loss
    # defalut parameters
    torch_default_dtype = torch.float32