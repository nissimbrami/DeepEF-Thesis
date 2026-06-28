import torch


class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cuda = device.type in ("cuda", "mps")
    debug = False
    if debug:
        data_path = './data/casp12_data_30/'
    else:
        data_path = './data/casp12_data_100/'
    inference_path = './data/inference_data'
    results_path = './res/results-emb/'
    seed = 42
    # Train data parameters
    homothresh = 0.9
    constraint = True
    split_train_size = 0.8
    debug_size  = 10
    sigma = 0.5 # for score matching loss (DSM D-only, 16 distance dims)
    # Model parameters
    h = 0.1
    coords_emb_size = 48
    embedding_size = 20
    filters = 64
    num_layers = 3
    dropout_rate = 0.2
    model_path = "./res/trianed_models-newDecoys/"
    light_attention = True
    gaussian_coef = -0.08
    # Embedding projection: "none", "mlp", "low_rank"
    emb_projection = "mlp"
    emb_proj_dim = 16        # output dim of projection (integrated into GNN)
    emb_proj_hidden = 128    # hidden dim for MLP projection
    emb_proj_rank = 4        # bottleneck rank for low_rank projection
    emb_input_dim = 1024     # LLM embedding dimension
    # Serial fusion: project PLM into GNN input so message-passing uses PLM signal
    serial_fusion = False
    serial_fusion_dim = 64   # projection dim for serial fusion path
    # Learned amino acid embeddings (replaces 20-dim one-hot with nn.Embedding)
    use_learned_aa = False
    aa_emb_dim = 64          # learned AA embedding dimension
    #training parameters
    lr = 0.0001
    wd = 0.00001
    batch_size = 1
    num_workers = 8
    persistent_workers = True   # avoid worker restart overhead each epoch
    prefetch_factor = 4         # pre-load batches in background
    N = 10
    num_epochs = 50
    seq_len = 450
    SM = False # score matching loss
    gradient_penalty = True
    decoy_threshold = 20
    max_grad_norm = 10.0
    clip_grad_norm = True
    reg_alpha = 0.1
    tau = 1.0 # temperature for InfoNCE contrastive loss
    gat_cutoff = 12.0 # Angstroms, distance cutoff for GAT edges (None = fully connected)
    compile_model = True  # torch.compile for kernel fusion (~10-30% speedup on PyTorch 2+)
    # defalut parameters
    torch_default_dtype = torch.float32
    precision = torch.float32