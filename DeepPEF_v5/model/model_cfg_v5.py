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

    # ==================================================================================
    # v5 proposal levers (A–F). Every default below reproduces the baseline tensor shapes
    # and energy math BIT-FOR-BIT. Each lever is opt-in via a CLI flag in pnas_train_v5.py,
    # which mutates the corresponding CFG field before the model/graph builder read it.
    # The whole point of the "dimension contract": graph builder and model read the SAME
    # numbers from THIS object, so widths never drift.
    # ==================================================================================
    # Shared distance-feature width. The graph layout is [D(dist_dim) | Fb(2*dist_dim) |
    # burial(burial_dim) | emb(E) | one_hot(20)]. dist_dim=16 is the historical value
    # (4 backbone atoms => 4*4=16 atom-atom distance channels).
    dist_dim = 16
    # Lever B — RBF kernel bank. rbf_centers=0 => single Gaussian kernel (exp(coef*D^2)),
    # exactly the original. rbf_centers=M>0 => sum of M Gaussians centered on a grid in
    # [rbf_min, rbf_max]; the result is summed back to width dist_dim (dimension-preserving).
    rbf_centers = 0
    rbf_min = 0.0
    rbf_max = 20.0
    # Lever A — energy decomposition. energy_terms=1 => a single per-residue energy scalar
    # (identical to baseline). K>1 => the final head outputs K per-residue terms which are
    # summed (after optional length_norm) into the total energy; per-term values are stored
    # for interpretability.
    energy_terms = 1
    # Lever C — burial/solvation feature. use_burial=False + burial_dim=0 => no extra node
    # feature (baseline). When on, a per-residue CB neighbor-density scalar is inserted BEFORE
    # the embedding block; burial_dim is the width it occupies (1).
    use_burial = False
    burial_dim = 0
    burial_radius = 10.0
    # Lever D — Flory unfolded reference. flory_unfolded=False => the unfolded graph zeros all
    # off-tridiagonal contacts (baseline). When on, the unfolded distances follow a random-coil
    # scaling |i-j|^nu instead, a physics-grounded reference state.
    flory_unfolded = False
    flory_nu = 0.5
    # Lever D+ — AFRC sequence-specific unfolded reference. flory_afrc=False => the analytic
    # b*|i-j|^nu coil (above). flory_afrc=True (requires flory_unfolded=True) => use the
    # Analytical Flory Random Coil (idptools/afrc) ensemble-average inter-residue distance map
    # for THIS sequence as the unfolded reference — the same random-coil-distogram idea IFUM
    # (Lee et al., Nat. Commun. 2026) used to reach SOTA. Value-only: output shape unchanged.
    # If the `afrc` package is not installed, the builder falls back to the analytic coil, so
    # the flag is safe to leave in configs even without the dependency.
    flory_afrc = False
    # Lever E — decoy denoising auxiliary head. denoise_weight=0 => head absent, no aux loss
    # (baseline). w>0 => an MLP off the 128-dim pre-energy features predicts injected coord
    # noise; a w*MSE term is added to the TRAIN loss only (never validation).
    denoise_weight = 0.0
    denoise_sigma = 0.3
    denoise_prob = 0.5
    # Lever F — extended GCN connectivity + edge features. gcn_span=1 => sequential (i,i+1)
    # edges only (baseline). S>1 => also connect offsets 2..S both directions. use_edge_features
    # => GATv2 layers consume a 41-dim edge_attr [onehot_src(20)|onehot_dst(20)|dist(1)].
    gcn_span = 1
    use_edge_features = False
    # Lever G — antisymmetry + mutation-delta (the JanusDDG lever). All default OFF = baseline.
    #   mutation_delta: concatenate (emb_mut - emb_wt) as an extra node block so the model sees
    #     WHAT CHANGED, not just the whole mutant embedding. Adds mutation_delta_dim node width.
    #   antisymmetry_weight: w>0 adds a loss term enforcing ddG(A->B) = -ddG(B->A) using the
    #     reverse-mutation pass (train only). w=0 => no reverse pass, exact baseline.
    #   reverse_mut_prob: probability of building the reverse mutation for the antisymmetry term.
    mutation_delta = False
    mutation_delta_dim = 0
    antisymmetry_weight = 0.0
    reverse_mut_prob = 0.5