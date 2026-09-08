"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, BatchNorm, TransformerConv

# W7 (edge attributes): scripts/ is not an importable package from every entry point,
# so locate edge_features.py relative to this file and add it to sys.path. A failed
# import must NOT break the default path, which never touches these symbols.
try:
    import os as _os, sys as _sys
    _sd = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'scripts')
    if _sd not in _sys.path:
        _sys.path.append(_sd)
    import edge_features as _EF
except Exception:
    _EF = None
from model.model_cfg import CFG
# U5/U6: coil-consistent edge topology + per-half CA coordinates. Flat import, matching
# the tree's convention for integrated modules (train_utils does the same with
# aa_descriptors). Every helper below defaults to CURRENT behaviour and returns the
# IDENTICAL input object when --coil_edges is off, so the off-path is byte-identical by
# construction rather than by reconstruction.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import coil_topology as _CT
# import matplotlib.pyplot as plt

class params():
    def __init__(self, embedding_size, layers, filters, cord_size, h, device):
        self.embedding_size = embedding_size
        self.layers = layers
        self.filters = filters
        self.cord_size = cord_size
        self.h = h
        self.device = device


class ProteinEnergyNet(nn.Module):
    """
    The neural network.
    """

    def __init__(self, params, name='ProteinEnergyNet'):
        """
        In the constructor we instantiate GNN layers and assign them as member variables.
        inputs: 
            params: embedding_size, layers,cord_size,h 
            name: name of the network
        """
        super(ProteinEnergyNet, self).__init__()
        self.name = name
        self.device = params.device
        # GNN parameters
        self.num_layers = params.layers
        self.n_filters = params.filters
        self.cord_size = params.cord_size
        self.n_atom_dist = 16
        self.emmbeding_size = params.embedding_size
        self.alpha = 0.1
        self.bonded = 1
        # dirivative error
        self.h = params.h
        # embedding params
        self.stdv = 1e-3
        self.Kembeddings = nn.Parameter(self.stdv * torch.randn(20, self.emmbeding_size, 9))
        # coordinate embedding parameters
        sigma = 1 + torch.zeros(3 * self.n_atom_dist, self.n_atom_dist, 5, 5)
        self.sigma = nn.Parameter(sigma)
        self.biasDistance = nn.Parameter(0.6 * torch.ones(1, 3 * self.n_atom_dist, 1, 1))
        self.KcoordsIn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3, self.cord_size)))  # 3 for x,y,z
        self.KcoordsOut = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.cord_size, 1)))

        # GNN layers  - each layes contains the params for matrix multiplication(TODO: what is the benefit in convolution)
        self.Kbond_layers = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_layers, self.n_filters,
                                                                            3 * self.n_atom_dist + self.emmbeding_size,
                                                                            self.bonded)))
        self.Knonbond_layers = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_layers, self.n_filters,
                                                                               3 * self.n_atom_dist + self.emmbeding_size,
                                                                               11)))

    def forward(self, X_decoy, X_native, embeiddng, emb_decoy):
        """
        This is where we define the network's forward pass, i.e. how the network maps inputs to outputs.
        The forward pass wiill recive the input data as a tensor.
        Inputs:
            X_decoy: a [batch_size,n_nodes ,num_atoms=4,coordination=3] tensor
            X_native:  a [batch_size,n_nodes ,num_atoms=4,coordination=3] tensor
            emmbeidng: a [batch_size,n_nodes, embedding_size] tensor
            emb_decoy: a [batch_size,n_nodes, embedding_size] tensor
        Since every node is connected to all other nodes there are no need for ajacency matrix.
        Returns:
            Energy [batch_size] tensor.
        """

        # Calculate energy for decoy and native
        E_xd = self.forward_x(X_decoy, emb_decoy)
        E_xn = self.forward_x(X_native, embeiddng)
        # Concatenate the energy of the decoy and native
        E_xd = E_xd.unsqueeze(1)
        E_xn = E_xn.unsqueeze(1)
        return torch.cat((E_xd, E_xn), dim=1)

    def forward_x(self, X, embeiddng):
        """
        Recives a single protein and calculate the energy
        Args:
            X (torch.tensor): Batch of proteins [batch_size,n_nodes ,num_atoms=4,coordination=3]
            emmbeidng (_type_): Batch of proteins [batch_size,n_nodes, embedding_size]

        Returns:
            E torch.tensor : Batch of proteins energy [batch_size]
        """
        B, n_residue, n_atoms, n_coords = X.shape
        # X_embed = self.embed_coords(X) # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        # X_centered = X-X.mean(dim=1, keepdim=True)
        # X_embed = X_centered
        X_embed = X
        Fh, A, G = self.get_Fh0(X_embed, embeiddng, self.h)  # [batch_size, n_nodes ,atom_dist+embedding_size]
        B, N, d = Fh.shape
        # Start GNN layers loop:
        for layer in range(self.num_layers):
            # calculate average and gradient of each neighbor
            Ki = self.Knonbond_layers[layer]
            Ki_hat = self.Kbond_layers[layer]
            # Get new average and gradient of each node
            A, G = self.get_AVG_mat(Fh), self.get_grad_mat(Fh)
            # Generate Fhb for bonded atoms
            Fhb = torch.zeros(B, n_residue, self.emmbeding_size + 3 * n_atoms ** 2, device=self.device)
            for i in range(self.bonded, Fh.shape[1], self.bonded):
                Fhb[:, (i - self.bonded):i, :] = self.layer_operation(Ki_hat,
                                                                      A[:, (i - self.bonded):i, (i - self.bonded):i],
                                                                      G[:, (i - self.bonded):i, (i - self.bonded):i],
                                                                      Fh[:, (i - self.bonded):i, :])
            # Generate Fhub for noneboned atoms
            Fhub = self.layer_operation(Ki, A, G, Fh)
            # Update Feature vector for each node
            Fh = Fh - self.alpha * Fhub - self.alpha * Fhb
            Fh = F.normalize(Fh, p=2, dim=1)
            # Calculate energy
        E = self.get_energy(Fh)

        return E

    def get_energy(self, Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [batch_size, n_nodes , embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sum(Fh ** 2, dim=(1, 2))
        return E

    def embed_cords(self, X_decoy):
        """
        Embeds the item into a vector representation.
        Inputs:
            X_decoy: a [batch_size, n_nodes ,num_atoms=4,coordination=3] tensor
        Returns:
            X: a [batch_size, n_nodes ,num_atoms=4,new_cords_size, embedding_size] tensor
        
        3.1 equation from the research paper
        """
        X_centered = X_decoy - X_decoy.mean(dim=1, keepdim=True)
        X = torch.matmul(X_centered ** 2, self.KcoordsIn)  # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        X = F.relu(X)
        X = torch.matmul(X, self.KcoordsOut)  # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        return X * X_centered

    def layer_operation(self, Ki, A, G, Fh):
        """
        Return the node features
        Args:
            K (tensor): weight matrix [n_filters,param1, param2]
            A (tensor): [batch_size, n_nodes, n_nodes] - avrege of each node
            G (tensor): [batch_size, n_nodes, n_nodes] - gradient of each node
            Fh (tensor): [batch_size,n_nodes, d] - node features

        Returns:
            tensor : [batch_size,n_nodes, embedding_size+n_nodes]
        """
        B, N_residu, _ = A.shape
        nodeE = Fh
        Q = torch.matmul(A, nodeE) + torch.matmul(G, nodeE)  # [batch_size,n_nodes,d]
        # Change shape to fit the conv1d
        Q = Q.reshape(B, -1, N_residu)  # [batch_size,d,n_nodes]
        Q = F.conv1d(Q, Ki)
        # Q = F.instance_norm(Q)
        Q = F.leaky_relu(Q, negative_slope=0.2)
        Q = F.conv_transpose1d(Q, Ki)
        Q = Q.reshape(B, N_residu, -1)  # [batch_size, n_nodes, filters]
        Q = torch.matmul(A, Q) + torch.matmul(G, Q)  # [batch_size,n_nodes,embedding_size+n_nodes]
        return Q

    def get_Fh0(self, Xd, FS, h):
        """
        Return the node features
        Args:
            Xd (tensor):X decoy [batch_size, n_nodes ,num_atoms=4,new_cords_size]
            FS (_type_): node features [batch_size,n_nodes, embedding_size]
            h (_type_): derovative step

        Returns:
            Fh (tensor): [batch_size,n_nodes, embedding_size+atom_dist]
            A (tensor): [batch_size, n_nodes, n_nodes]
            G (tensor): [batch_size, n_nodes, n_nodes]
        """
        B, N_residu, N_atoms, coords_size = Xd.shape
        D = self.get_dist_matrix(
            Xd)  # [batch_size, n_nodes,n_nodes, atom_dist=16]-> [batch_size,n_nodes*atom_dist, n_nodes]
        # Compute the gussian of the distance matrix
        D = torch.swapaxes(torch.swapaxes(D, 3, 2), 2,
                           1)  # D.reshape(B,N_atoms**2,N_residu,N_residu)              # [batch_size, n_nodes*atom_dist,n_nodes]
        Z = F.conv2d(D, self.sigma.abs(), padding=self.sigma.shape[-1] // 2)
        Z = F.normalize(Z, dim=[2, 3])
        D = torch.relu(torch.exp(-1e1 * Z) - self.biasDistance)
        # Sum for each atom 16 distances
        D = D.sum(dim=2)  # [batch_size, n_nodes,atom_dist=16]
        D = F.normalize(D, p=2, dim=1)  # [batch_size,n_nodes, atoms_dist=16]
        D = torch.swapaxes(D, 1, 2)  # D.reshape(B,N_residu,-1)
        # Get the derivative of the distance matrix
        G = self.get_grad_mat(D)  # [batch_size, n_nodes,n_nodes]
        # Get the average of the distance matrix
        A = self.get_AVG_mat(D)  # [batch_size, n_nodes, n_nodes]
        # First node features
        FD = torch.matmul(A, D) + torch.matmul(G, D)  # [batch_size,n_nodes, atom_dist=16]
        # TODO: differences between atoms of the same node are small
        FD = F.normalize(FD, p=2, dim=1)  # [batch_size,n_nodes, atoms_dist=16]
        Fh = torch.cat((FD, FS), dim=2)  # [batch_size,n_nodes, embedding_size+atoms_dist=16]
        return Fh, A, G

    def get_dist_matrix(self, Xd):
        """
        Return the node distence matrix
        Args:
            Xd (tensor):X embeded [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        Returns:
            tensor : [batch_size, n_nodes,n_nodes ,atom_dist=16] tensor
        """
        B, n_residue, N_atoms, coords_size = Xd.shape
        Xd = Xd.reshape(B, n_residue * N_atoms, coords_size)
        D = torch.cdist(Xd, Xd, p=2)
        D = D.reshape(B, n_residue, N_atoms, n_residue, N_atoms)
        D = torch.swapaxes(D, 2, 3)
        D = D.reshape(B, n_residue, n_residue, N_atoms * N_atoms)
        return D

    def get_AVG_mat(self, Fh):
        """
        Return the node distence matrix between all nodes

        Args:
            Fh (tensor): tensor of node features [batch_size,n_nodes, embedding_size+n_nodes*atom_dist]
            
        output:
            AVG_MAT (tensor) : [batch_size,n_nodes, n_nodes] tensor
        """
        B, n_residue, _ = Fh.shape
        # Calculate the pairwise avrege between each node in the tensor
        pairwise_avg = (Fh.unsqueeze(axis=2) + Fh.unsqueeze(axis=1)) / 2
        return torch.sum(pairwise_avg, axis=-1)

    def get_grad_mat(self, Fh):
        """
        Return the node distence matrix between all nodes

        Args:
            Fh (tensor): tensor of node features [batch_size,n_nodes, embedding_size+n_nodes*atom_dist]
            
        output:
            Grad_MAT (tensor) : [batch_size,n_nodes, n_nodes] tensor
        """
        # Get the number of nodes in each batch and the dimensionality of each node
        batch_size, n_nodes, d_dims = Fh.shape
        # Calculate the pairwise differences between each node in the tensor
        pairwise_differences = (Fh.unsqueeze(axis=2) - Fh.unsqueeze(axis=1)) / self.h
        # Calculate the pairwise squared distances between each node in the tensor
        # pairwise_squared_distances = torch.sum(pairwise_differences**2, axis=-1)
        # # Calculate the pairwise distances between each node in the tensor
        # distances = torch.sqrt(pairwise_squared_distances)
        return torch.sum(pairwise_differences, axis=-1)


class MLPProjection(nn.Module):
    """MLP projection: 1024 -> hidden -> proj_dim"""
    def __init__(self, input_dim=1024, hidden_dim=128, output_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class LowRankProjection(nn.Module):
    """Low-rank factored projection: 1024 -> rank -> proj_dim"""
    def __init__(self, input_dim=1024, rank=4, output_dim=16):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim)
    def forward(self, x):
        return self.up(self.down(x))


def _aa_desc_mode(cfg):
    """W6: the active descriptor mode, or 'none' if the module is absent.
    A flag set while the module is missing is a HARD ERROR, never a silent fallback --
    a run that quietly ignored --aa_descriptors would be reported as a descriptor
    result and would be a lie."""
    requested = getattr(cfg, 'aa_descriptors', 'none') or 'none'
    try:
        from aa_descriptors import aa_descriptor_mode
    except ImportError:
        if requested != 'none':
            raise ImportError(
                "W6: --aa_descriptors %s was requested but aa_descriptors.py is not "
                "importable. Copy it to the repo root beside train_utils.py." % requested)
        return 'none'
    return aa_descriptor_mode(cfg)


def _aa_desc_dim(mode, cfg):
    """W6: K, read from the committed CSV. 0 when off."""
    if mode == 'none':
        return 0
    from aa_descriptors import descriptor_dim
    return descriptor_dim(mode, cfg)


def _ligand_block_dim(cfg):
    """W11's block width, 10 when --ligand_nodes is on, else 0.

    Guarded import: ligand_features.py lives in scripts/ and W11 is optional. Absent
    module => 0 => byte-identical to the pre-W11 tree.
    """
    if not getattr(cfg, 'ligand_nodes', False):
        return 0
    try:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(
            _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), 'scripts'))
        from ligand_features import LIGAND_DIM
        return LIGAND_DIM
    except Exception:                                        # noqa: BLE001
        raise RuntimeError(
            '--ligand_nodes requires scripts/ligand_features.py, which could not be '
            'imported. Refusing to run with a silently-zero-width block.')


def _sidechain_block_dim(cfg):
    """W12 side-chain block width: 4 when --sidechain_features, else 0.

    Delegates to scripts/sidechain_features.sidechain_dim so the width has ONE
    authority -- the same discipline W6's descriptor K follows.
    """
    if not getattr(cfg, 'sidechain_features', False):
        return 0
    from sidechain_features import sidechain_dim
    return sidechain_dim(cfg)


def _sibling_block_dims(cfg):
    """Width of the OTHER optional blocks that sit between W6 and W11: W9 then W8.

    W11 is the last block before emb, so its start must count every block ahead of it.
    Both are guarded: either sibling lever may be absent from a given tree. If this
    number is wrong the model reads the wrong columns WITHOUT crashing, which is why
    scripts/gate_ligand.py gate E checks all sixteen on/off combinations.
    """
    total = 0
    if getattr(cfg, 'metal_features', False):
        # W9 IS RETIRED. train_utils.get_graph concatenates NO metal block, so adding
        # METAL_DIM here would widen fc1 and shift the W11 ligand block 9 columns into
        # the LLM embedding -- a silent wrong-column read, not a crash. Refuse instead
        # of quietly returning 9. See scripts/metal_features.py._w9_retired.
        raise RuntimeError(
            "W9 --metal_features is RETIRED (superseded by W11 --ligand_nodes). "
            "hydro_net._sibling_block_dims() refuses to reserve METAL_DIM=9 columns "
            "for a block that train_utils.get_graph never assembles: doing so would "
            "shift the W11 ligand slice 9 columns into the LLM embedding WITHOUT "
            "crashing. There is no --metal_features flag in train.py; do not add one. "
            "Use --ligand_nodes with --ligand_annotations instead.")
    if getattr(cfg, 'struct_quality', False):
        try:
            from struct_quality import STRUCT_QUALITY_DIM
            total += STRUCT_QUALITY_DIM
        except Exception:                                    # noqa: BLE001
            total += 3
    # W12 IS wired: train_utils.get_graph concatenates the 4-column side-chain block
    # between W6 and W11, so unlike the RETIRED W9 its width is genuinely present in
    # the feature vector and MUST be counted here -- otherwise --ligand_nodes would
    # slice four columns early and read the wrong data WITHOUT crashing.
    total += _sidechain_block_dim(cfg)
    return total


class PEM(torch.nn.Module):
    """Protein energy model"""

    def __init__(self, layers, gaussian_coef,dropout_rate = 0.2, light_attention=False, emb_projection="none", gat_cutoff=None, readout='sum'):
        super().__init__()
        # Experiment 1: learned aggregation of per-residue energies instead of the plain sum.
        #   'sum'       : G = sum_i e_i               (default, extensive energy — baseline)
        #   'attention' : G = N * sum_i softmax(s_i) e_i   (learned per-residue weights)
        #   'gated'     : G = sum_i sigmoid(s_i) e_i        (learned per-residue gates, keeps extensivity)
        self.readout = readout

        # Embedding projection config: "none", "mlp", "low_rank"
        self.emb_projection_type = emb_projection
        emb_proj_dim = CFG.emb_proj_dim  # 16

        if emb_projection == "mlp":
            self.emb_projector = MLPProjection(
                input_dim=CFG.emb_input_dim, hidden_dim=CFG.emb_proj_hidden,
                output_dim=emb_proj_dim, dropout=dropout_rate)
        elif emb_projection == "low_rank":
            self.emb_projector = LowRankProjection(
                input_dim=CFG.emb_input_dim, rank=CFG.emb_proj_rank,
                output_dim=emb_proj_dim)
        else:
            self.emb_projector = None

        # Dimensions depend on whether embeddings are projected into GNN
        # When projected: emb_proj_dim added to GNN input, not concatenated after
        proj_extra = emb_proj_dim if self.emb_projector is not None else 0
        post_gnn_emb = 0 if self.emb_projector is not None else CFG.emb_input_dim

        # GCN layers
        gcn_dim_in = 36 + proj_extra  # 36 or 52
        gcn_dim_h = 64
        gcn_dim_out = 36 + proj_extra
        self.graph_model_gcn = [GCN(gcn_dim_in, gcn_dim_h, gcn_dim_out, dropout_rate) for i in range(layers)]
        # GAT layers
        gat_dim_in = 36 + proj_extra  # 36 or 52
        gat_dim_h = 64
        gat_dim_out = 36 + proj_extra
        # W7: pairwise edge attributes = src one-hot(20) + dst one-hot(20) + 16 RBF = 56,
        # handed to GATv2Conv as edge_dim. OFF (the default) leaves edge_dim None, so the
        # GAT constructor call is the historical one and no edge parameters are created.
        self.edge_features = bool(getattr(CFG, 'edge_features', False))
        if self.edge_features and _EF is None:
            raise RuntimeError('--edge_features requires scripts/edge_features.py, which '
                               'could not be imported.')
        self.edge_attr_dim = _EF.edge_attr_dim() if self.edge_features else None
        self.graph_model_gat = [GAT(gat_dim_in, gat_dim_h, gat_dim_out, 8, dropout_rate,
                                    edge_dim=self.edge_attr_dim) for i in range(layers)]
        # Gaussian coefficient
        self.gaussian_coef = gaussian_coef
        # graph attention layers
        self.GAT_layers = torch.nn.ModuleList(self.graph_model_gat)
        self.GCN_layers = torch.nn.ModuleList(self.graph_model_gcn)
        # Fully connected layers - GCN
        # W5: the solvation block (burial, hydropathy, burial*hydropathy) sits between
        # Fb and emb, so LEFT-anchored widths grow and RIGHT-anchored indices do not.
        self.solv_dim = 3 if getattr(CFG, 'burial_features', False) else 0
        self.solv_start = 48                      # after D(16) + Fb(32)
        _sd = self.solv_dim
        # W6: the descriptor block sits immediately after the solvation block, so it
        # starts at 48 + solv_dim. K comes from the committed CSV, never hard-coded.
        self.desc_mode = _aa_desc_mode(CFG)
        self.desc_dim = _aa_desc_dim(self.desc_mode, CFG)
        self.desc_start = self.solv_start + self.solv_dim
        _dd = self.desc_dim
        # W11: the ligand contact block sits after every other new block and before emb,
        # so LEFT-anchored widths grow and the RIGHT-anchored indices do not move. Its
        # start counts W5 solv, W6 desc, and (when those sibling levers are applied)
        # W9 metal and W8 confidence -- see ligand_features.ligand_start, which is the
        # ONE authority for this arithmetic; a mismatch here would silently feed the
        # wrong columns into fc1 and never crash.
        # W12: the side-chain chemistry block sits immediately after the W6 descriptor
        # block, at 48 + solv_dim + desc_dim, and BEFORE the W11 ligand block. Its width
        # is counted by _sibling_block_dims so lig_start moves with it; without that the
        # ligand slice would start 4 columns early and read them without crashing.
        self.sc_dim = _sidechain_block_dim(CFG)
        self.sc_start = self.desc_start + self.desc_dim
        _cd = self.sc_dim
        self.lig_dim = _ligand_block_dim(CFG)
        self.lig_start = self.desc_start + self.desc_dim + _sibling_block_dims(CFG)
        _ld = self.lig_dim
        # ONLY fc1_* grow. fc2_* project back to FIXED internal widths, so inst_norm1,
        # inst_norm2 and fc_in_dim must NOT change -- widening them was a real bug.
        self.fc1_gcn = nn.Linear(52 + _sd + _dd + _cd + _ld + proj_extra, 64) # 52 = 32(dist) + 20(one-hot) [+ solv] [+ desc] [+ sidechain] [+ lig] [+ proj_extra]
        self.fc2_gcn = nn.Linear(64, gcn_dim_in)
        # Fully connected layers - GAT
        self.fc1_gat = nn.Linear(36 + _sd + _dd + _cd + _ld + proj_extra, 64) # 36 = 16(dist) + 20(one-hot) [+ solv] [+ desc] [+ sidechain] [+ lig] [+ proj_extra]
        self.fc2_gat = nn.Linear(64, gat_dim_in)
        # normalization layers
        self.inst_norm1 = Normalization_layer(36 + proj_extra, affine=True)
        self.inst_norm2 = Normalization_layer(2 * (36 + proj_extra), affine=True)
        # Fc layers for the final output
        fc_in_dim = 2 * (36 + proj_extra) + post_gnn_emb  # 72+1024 (none) or 104+0 (projected)
        self.fc1 = nn.Linear(fc_in_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        if readout in ('attention', 'gated'):
            self.readout_score = nn.Linear(128, 1)   # per-residue weight/gate logit

        # energy epsilon
        self.energy_epsilon = 1

        # embedding indexes
        self.one_hot_index = -20
        self.bonded_index = 48
        self.non_bonded_index = 16
        self.llm_index = -(CFG.emb_input_dim + 20)  # dynamic based on embedding dim

        # edge index cache
        self._edge_cache_key = None
        self._edge_cache = None

        # GAT distance cutoff (Angstroms); None = fully connected
        self.gat_cutoff = gat_cutoff

        # light attention machanism
        self.light_attention = light_attention
        if self.light_attention:
            self.LA = LightAttention(embeddings_dim=fc_in_dim)
        
    
    def forward(self,x,f_type = 'Default', ca_coords=None, n_folded=None):
        """
                Forward function
             Args:
            x (tensor): [batch, n_nodes, bonded_features+non_bonded_features+LLM_features]
            f_type (str, optional): 'A_inference' or 'defualt', if 'A_inferece' return each amino acid energy . Defaults to 'Default'.
            ca_coords (tensor, optional): [batch, n_nodes, 3] CA atom coordinates for distance-based GAT edges.
            n_folded (int, optional): U5/U6 -- how many LEADING rows of the batch are the
                FOLDED state. train.py concatenates [folded; unfolded] along the batch dim
                and makes ONE model call, so the split is a caller convention and nothing
                in the tensor marks it. It is passed EXPLICITLY and never inferred as B//2:
                get_ddg_head calls this with a FOLDED-ONLY batch under f_type='features',
                and inferring B//2 there would declare half a folded batch unfolded and
                silently corrupt it. None == 'every row is folded' == today's behaviour.

        Returns:
            if f_type == 'A_inference':
                energy: native and decoy energy for each amino acid
            if f_type == 'Default':
            energy: native and decoy energy
        """
        # Get the edge index
        # U6 runs BEFORE the edge build so a CA-cutoff topology is derived from the
        # per-half coordinates, not from folded coordinates broadcast over both halves.
        ca_coords = _CT.per_half_ca_coords(ca_coords, x.shape[0], x.shape[1], n_folded, CFG)
        edge_index_gcn,edge_index_gat = self.get_edge_index(x, ca_coords=ca_coords,
                                                            n_folded=n_folded)
        # reshape x to [batch_size*n_nodes,features]
        B, N, _ = x.shape
        x = x.reshape(B * N,-1)
        # split features to 2 graphs, bonded and non-bonded
        # W5 + W6: extend this branch, never bypass it. NOTE the GCN slice is x[:, :32]
        # -- D(16) plus only HALF of Fb. That is the model as trained; widening it to
        # :48 feeds 71 dims into a 55-dim layer. Do not "fix" it here.
        _extra = []
        if self.solv_dim:
            _extra.append(x[:, self.solv_start:self.solv_start + self.solv_dim])
        if getattr(self, 'desc_dim', 0):
            _extra.append(x[:, self.desc_start:self.desc_start + self.desc_dim])
        # W11: the ligand block is LAST among the new blocks, matching the _blocks order
        # in train_utils.get_graph. Appended here, never spliced in earlier, so W5/W6
        # keep reading exactly the columns they read before this lever existed.
        # W12: after W6 desc and before W11 lig, matching the _blocks order in
        # train_utils.get_graph. NOTE the GCN branch's distance half is x[:, :32], so
        # this block reaches both branches ONLY through _extra.
        if getattr(self, 'sc_dim', 0):
            _extra.append(x[:, self.sc_start:self.sc_start + self.sc_dim])
        if getattr(self, 'lig_dim', 0):
            _extra.append(x[:, self.lig_start:self.lig_start + self.lig_dim])
        if _extra:
            x_gcn = torch.cat((x[:,:self.non_bonded_index + self.non_bonded_index], *_extra, x[:,self.one_hot_index:]),dim=-1)
            x_gat = torch.cat((x[:,:self.non_bonded_index], *_extra, x[:,self.one_hot_index:]),dim=-1)
        else:
            x_gcn = torch.cat((x[:,:self.non_bonded_index+ self.non_bonded_index],x[:,self.one_hot_index:]),dim=-1) # B*N,52
            x_gat = torch.cat((x[:,:self.non_bonded_index],x[:,self.one_hot_index:]),dim=-1) # B*N,36
        x_emb_features = x[:,self.llm_index:self.one_hot_index] # B*N,1024

        # Project embeddings and concatenate into GNN input, or keep for post-GNN concat
        if self.emb_projector is not None:
            x_proj = self.emb_projector(x_emb_features) # B*N,1024 -> B*N,proj_dim
            x_gcn = torch.cat((x_gcn, x_proj), dim=-1) # B*N, 52+proj_dim
            x_gat = torch.cat((x_gat, x_proj), dim=-1) # B*N, 36+proj_dim

        # forward pass through the graph attention and convolution layers
        # W7: build the [E,56] edge attributes from the RIGHT-ANCHORED one-hot block
        # x[:, -20:], which no left-anchored W5/W6 block can shift. OFF => None, and
        # forward_gat then runs the historical path untouched.
        edge_attr_gat = None
        if self.edge_features:
            edge_attr_gat = _EF.build_edge_attr(edge_index_gat, x[:, -20:], ca_coords, N)
        x1 = self.forward_gcn(x_gcn, edge_index_gcn, B, N) # B*N,gcn_in -> B*N,gcn_out
        x2 = self.forward_gat(x_gat, edge_index_gat, B, N, edge_attr=edge_attr_gat) # B*N,gat_in -> B*N,gat_out
        # concat features
        x = torch.cat((x1,x2),dim=-1) # B*N, gcn_out+gat_out
        # reshape to use instance norm
        x = x.reshape(B, N,-1)
        x = self.inst_norm2(x)
        x = x.reshape(B * N,-1)
        # Add raw LLM features only when no projection (original behavior)
        if self.emb_projector is None:
            x = torch.cat((x,x_emb_features),dim=-1) # B*N,72+1024->B*N,1096
        # Light attention machanism
        if self.light_attention:
            x = x.reshape(B, N,-1)
            x = x.swapaxes(1,2)
            x = self.LA(x)
            x = x.swapaxes(1,2)
            x = x.reshape(B * N,-1)
        # fc layers
        h = F.relu(self.fc1(x))      # [B*N, 128] per-residue feature
        x = self.fc2(h)              # [B*N, 1]   per-residue energy e_i
        x = x.reshape(B, N, 1)
        # Learned aggregation (Experiment 1): reweight per-residue energies before summing.
        if self.readout in ('attention', 'gated'):
            s = self.readout_score(h).reshape(B, N, 1)
            if self.readout == 'attention':
                x = torch.softmax(s, dim=1) * x * N    # weighted; *N keeps magnitude ~ a plain sum
            else:  # gated
                x = torch.sigmoid(s) * x
        if (f_type == 'Default'):
            return self.get_energy(x)
        elif(f_type == 'A_inference'): # return the (post-readout) per-residue energy
            return x
        elif(f_type == 'features'): # return per-residue latent features h [B,N,128] for a direct-ddG head
            return h.reshape(B, N, -1)
        
    def forward_gat(self, x, edge_index_gat, B, N, edge_attr=None):
        """forward function for the graph model"""
        if (getattr(self, 'solv_dim', 0) or getattr(self, 'desc_dim', 0)
                or getattr(self, 'sc_dim', 0) or getattr(self, 'lig_dim', 0)):
            # W5 + W6: the input is wider than fc2_gat's fixed output, so the residual
            # must be taken AFTER the projection or h1 + identity is a shape error.
            # W6 extends this same branch rather than adding a second one: the real
            # condition is "is the input wider than baseline", and either block widens it.
            x = self.fc1_gat(x)
            x = F.relu(x)
            x = self.fc2_gat(x)
            identity = x
        else:
            identity = x # identity for the residual connection
            x = self.fc1_gat(x) # N,36->N,64
            x = F.relu(x)
            x = self.fc2_gat(x) # N,64->N,36
        # swap axis to use insrance norm
        x = x.reshape(B, N,-1)
        x = self.inst_norm1(x)
        x = x.reshape(B * N,-1)
        for gat_layer in self.GAT_layers:
            h1,z = gat_layer(x, edge_index_gat, B, N, edge_attr=edge_attr)
            x = h1 + identity

        return x

    def forward_gcn(self, x, edge_index_gcn, B, N):
        """forward function for the graph model"""
        x = self.fc1_gcn(x) # N,36->N,64
        x = F.relu(x)
        x = self.fc2_gcn(x) # N,64->N,36
        # swap axis to use insrance norm
        x = x.reshape(B, N,-1)
        x = self.inst_norm1(x)
        x = x.reshape(B * N,-1)
        identity = x # identity for the residual connection
        for gcn_layer in self.GCN_layers:
            h1,z = gcn_layer(x, edge_index_gcn, B, N)
            x = h1 + identity
        return x
  
    def get_energy(self,Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [n_nodes , embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sum(Fh,dim=(1,2))
        # E = torch.log(torch.sum(Fh,dim=(1,2)) + self.energy_epsilon)
        return E
  
    def get_edge_index(self, x, ca_coords=None, n_folded=None):
        """Return the edge index for the graph convolution and attention layers.
        The edge index of the gcn is a line from the amino acid to the next amino acid.
        The edge index of the gat uses a distance cutoff on CA atoms when ca_coords
        is provided and self.gat_cutoff is set; otherwise falls back to fully connected.
        Results are cached only for the fully-connected (shape-based) case."""
        B, N = x.shape[0], x.shape[1]

        # Cache only when edges depend solely on shape (no ca_coords)
        # U5: with --coil_edges the edge set additionally depends on WHERE the
        # folded/unfolded boundary sits and on whether the coil is on, so a (B, N) key
        # would hand a coil batch the folded-topology edge set out of the cache. That is a
        # correctness bug, not a performance nicety. edge_cache_key carries both, plus
        # gcn_span/gcn_bidir which are read off CFG at call time.
        if ca_coords is None:
            key = _CT.edge_cache_key(B, N, CFG, n_folded)
            if self._edge_cache_key == key and self._edge_cache is not None:
                return self._edge_cache

        # GCN: chain edges within each batch element.
        # W7: the baseline set is (i, i+1) DIRECTED FORWARD, so with three layers the
        # reach along the chain is three residues. An alpha-helix is defined by i->i+4
        # and a beta-sheet by i->i+2, so secondary structure is currently
        # unrepresentable. --gcn_span extends the offsets to |i-j| <= span.
        # U10: at span 1 the baseline is directed. Making all offsets bidirectional
        # while extending the span would change TWO things at once, so --gcn_bidir is
        # its own flag and can be set at span 1 to give an honest control arm.
        # Both default to the historical behaviour and are bit-identical when off.
        dev = x.device
        offsets = torch.arange(B, device=dev).unsqueeze(1) * N  # [B, 1]
        span = int(getattr(CFG, 'gcn_span', 1))
        bidir = bool(getattr(CFG, 'gcn_bidir', False))
        if span == 1 and not bidir:
            local_gcn = torch.arange(N - 1, device=dev)
            gcn_src = (local_gcn.unsqueeze(0) + offsets).reshape(-1)
            gcn_dst = gcn_src + 1
            edge_index_gcn_all = torch.stack([gcn_src, gcn_dst])
        else:
            src_parts, dst_parts = [], []
            for k in range(1, max(1, span) + 1):
                if N - k <= 0:
                    continue
                loc = torch.arange(N - k, device=dev)
                a = (loc.unsqueeze(0) + offsets).reshape(-1)
                b = a + k
                src_parts.append(a); dst_parts.append(b)
                if bidir:
                    src_parts.append(b); dst_parts.append(a)
            edge_index_gcn_all = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])

        # GAT: distance-cutoff or fully connected
        if ca_coords is not None and self.gat_cutoff is not None:
            # Distance-based edges: connect CA atoms within cutoff radius
            dists = torch.cdist(ca_coords, ca_coords)  # [B, N, N]
            mask = (dists < self.gat_cutoff) & (dists > 0)  # exclude self-loops
            batch_idx, src_idx, dst_idx = torch.where(mask)
            flat_src = batch_idx * N + src_idx
            flat_dst = batch_idx * N + dst_idx
            edge_index_gat_all = torch.stack([flat_src, flat_dst])
        else:
            # Fully connected within each batch element (original behavior)
            arange = torch.arange(N, device=dev)
            src, dst = torch.meshgrid(arange, arange, indexing='ij')
            mask = src != dst
            local_src, local_dst = src[mask], dst[mask]
            gat_src = (local_src.unsqueeze(0) + offsets).reshape(-1)
            gat_dst = (local_dst.unsqueeze(0) + offsets).reshape(-1)
            edge_index_gat_all = torch.stack([gat_src, gat_dst])

        # U5: replace the UNFOLDED rows' GAT topology with the bidirectional chain.
        # A random coil has bonded neighbours and nothing else, so the fully-connected /
        # CA-cutoff set is exactly the folded CONTACT TOPOLOGY leaking into the reference
        # state (defect A5). The GCN set is deliberately NOT touched: it is already
        # chain-local by construction (|i-j| <= gcn_span), carries no folded contacts, and
        # changing it too would confound U5 with W7/U10.
        # Off (default) this returns the IDENTICAL object -- byte-identical, and it also
        # RAISES if --coil_edges is set without --flory_unfolded rather than no-opping.
        edge_index_gat_all = _CT.apply_coil_edges(
            edge_index_gat_all, B, N, n_folded, dev, CFG)

        if ca_coords is None:
            self._edge_cache_key = key
            self._edge_cache = (edge_index_gcn_all, edge_index_gat_all)

        return edge_index_gcn_all, edge_index_gat_all


class GraphTransformerBlock(nn.Module):
    """Residual graph transformer block for residue-level energy features."""
    def __init__(self, hidden_dim, heads, edge_dim, dropout_rate):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.conv = TransformerConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout_rate,
            edge_dim=edge_dim,
            beta=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        x = self.norm1(x + self.dropout(h))
        h = self.ff(x)
        return self.norm2(x + self.dropout(h))


class PEMGraphTransformer(torch.nn.Module):
    """Protein energy model using Graph Transformer message passing."""
    def __init__(
        self,
        layers=None,
        gaussian_coef=None,
        dropout_rate=0.2,
        light_attention=False,
        emb_projection="mlp",
        gt_hidden_dim=None,
        gt_heads=None,
        gt_edge_cutoff=None,
        gt_edge_rbf_dim=None,
    ):
        super().__init__()
        # W6/W11 SILENT-NO-OP GUARD.  This architecture reassembles its node vector from
        # FIXED slices of flat_x: [:16] non-bonded, [16:48] bonded, [-1044:-20] emb and
        # [-20:] one-hot.  Those are anchored at BOTH ends, so inserting any optional
        # block in the middle (W5 solv at 48, W6 desc after it, W9/W8/W11 after that)
        # leaves every one of those slices IN BOUNDS and correctly shaped.  Nothing
        # raises, the inserted columns are simply never read, and the run completes and
        # reports a BASELINE number that would be written up as a descriptor/ligand
        # result.  That is the signature failure mode of this project, so refuse at
        # CONSTRUCTION -- before the dataset is loaded, not after an epoch of training.
        _blocked = []
        if _aa_desc_mode(CFG) != 'none':
            _blocked.append('--aa_descriptors %s (W6, K=%d)'
                            % (getattr(CFG, 'aa_descriptors', 'none'),
                               _aa_desc_dim(_aa_desc_mode(CFG), CFG)))
        if getattr(CFG, 'burial_features', False):
            _blocked.append('--burial_features (W5, 3 cols)')
        if getattr(CFG, 'metal_features', False):
            _blocked.append('--metal_features (W9)')
        if getattr(CFG, 'struct_quality', False):
            _blocked.append('--struct_quality (W8)')
        if getattr(CFG, 'ligand_nodes', False):
            _blocked.append('--ligand_nodes (W11, 10 cols)')
        if getattr(CFG, 'sidechain_features', False):
            _blocked.append('--sidechain_features (W12, 4 cols)')
        if _blocked:
            raise ValueError(
                "model_arch='graph_transformer' CANNOT read inserted feature blocks, but "
                "%s was requested.\n"
                "PEMGraphTransformer slices flat_x at fixed offsets ([:16], [16:48], "
                "[-1044:-20], [-20:]); an inserted block lands between 48 and -1044 and is "
                "silently DROPPED -- every slice stays in bounds, nothing raises, and the "
                "run would report a baseline number as a feature result.\n"
                "Use --model_arch pem (which sizes fc1_gcn/fc1_gat from the block widths), "
                "or teach PEMGraphTransformer the offsets before using it with this lever."
                % ' + '.join(_blocked))
        self.gaussian_coef = gaussian_coef
        self.hidden_dim = gt_hidden_dim or CFG.gt_hidden_dim
        self.heads = gt_heads or CFG.gt_heads
        self.layers = layers or CFG.gt_layers
        self.edge_cutoff = gt_edge_cutoff or CFG.gt_edge_cutoff
        self.edge_rbf_dim = gt_edge_rbf_dim or CFG.gt_edge_rbf_dim
        self.edge_dim = self.edge_rbf_dim + 2

        self.one_hot_index = -20
        self.llm_index = -1044
        self.non_bonded_index = 16
        self.bonded_dim = 32

        self.emb_projection_type = emb_projection
        if emb_projection == "mlp":
            self.emb_projector = MLPProjection(
                input_dim=CFG.emb_input_dim,
                hidden_dim=CFG.emb_proj_hidden,
                output_dim=CFG.emb_proj_dim,
                dropout=dropout_rate,
            )
            emb_dim = CFG.emb_proj_dim
        elif emb_projection == "low_rank":
            self.emb_projector = LowRankProjection(
                input_dim=CFG.emb_input_dim,
                rank=CFG.emb_proj_rank,
                output_dim=CFG.emb_proj_dim,
            )
            emb_dim = CFG.emb_proj_dim
        else:
            self.emb_projector = None
            emb_dim = CFG.emb_input_dim

        node_dim = self.non_bonded_index + self.bonded_dim + 20 + emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(self.hidden_dim, self.heads, self.edge_dim, dropout_rate)
            for _ in range(self.layers)
        ])
        self.node_norm = Normalization_layer(self.hidden_dim, affine=True)

        self.light_attention = light_attention
        if self.light_attention:
            self.LA = LightAttention(embeddings_dim=self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.energy_epsilon = 1

    def forward(self, x, f_type='Default', ca_coords=None, n_folded=None):
        B, N, _ = x.shape
        # U6: per-half coordinates, so the RBF edge features of the unfolded pass are not
        # the true folded CA distances. See PEM.forward for why n_folded is explicit.
        ca_coords = _CT.per_half_ca_coords(ca_coords, B, N, n_folded, CFG)
        flat_x = x.reshape(B * N, -1)

        dist_features = flat_x[:, :self.non_bonded_index]
        bonded_features = flat_x[:, self.non_bonded_index:self.non_bonded_index + self.bonded_dim]
        one_hot = flat_x[:, self.one_hot_index:]
        emb_features = flat_x[:, self.llm_index:self.one_hot_index]
        if self.emb_projector is not None:
            emb_features = self.emb_projector(emb_features)

        h = torch.cat([dist_features, bonded_features, emb_features, one_hot], dim=-1)
        h = self.input_proj(h)
        edge_index, edge_attr = self.get_edge_index_and_attr(x, ca_coords, n_folded=n_folded)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)

        h = h.reshape(B, N, -1)
        h = self.node_norm(h)
        if self.light_attention:
            h = h.swapaxes(1, 2)
            h = self.LA(h)
            h = h.swapaxes(1, 2)

        h = h.reshape(B * N, -1)
        h = F.relu(self.fc1(h))
        h = self.fc2(h).reshape(B, N, 1)
        if f_type == 'Default':
            return self.get_energy(h)
        elif f_type == 'A_inference':
            return h
        raise ValueError(f"Unsupported f_type: {f_type}")

    def get_energy(self, Fh):
        return torch.sum(Fh, dim=(1, 2))

    def get_edge_index_and_attr(self, x, ca_coords=None, n_folded=None):
        B, N = x.shape[0], x.shape[1]
        device = x.device
        offsets = torch.arange(B, device=device).unsqueeze(1) * N
        local_src_parts = []
        local_dst_parts = []

        if ca_coords is not None:
            dists = torch.cdist(ca_coords, ca_coords)
            mask = (dists < self.edge_cutoff) & (dists > 0)
            batch_idx, src_idx, dst_idx = torch.where(mask)
            src = batch_idx * N + src_idx
            dst = batch_idx * N + dst_idx
            local_src_parts.append(src)
            local_dst_parts.append(dst)
        else:
            arange = torch.arange(N, device=device)
            src_grid, dst_grid = torch.meshgrid(arange, arange, indexing='ij')
            mask = src_grid != dst_grid
            src = (src_grid[mask].unsqueeze(0) + offsets).reshape(-1)
            dst = (dst_grid[mask].unsqueeze(0) + offsets).reshape(-1)
            local_src_parts.append(src)
            local_dst_parts.append(dst)

        if N > 1:
            seq = torch.arange(N - 1, device=device)
            seq_src = torch.cat([seq, seq + 1])
            seq_dst = torch.cat([seq + 1, seq])
            seq_src = (seq_src.unsqueeze(0) + offsets).reshape(-1)
            seq_dst = (seq_dst.unsqueeze(0) + offsets).reshape(-1)
            local_src_parts.append(seq_src)
            local_dst_parts.append(seq_dst)

        edge_index = torch.stack([torch.cat(local_src_parts), torch.cat(local_dst_parts)], dim=0)
        # U5: swap the unfolded rows onto the chain-local set BEFORE the unique(), so the
        # sequence edges appended above are deduplicated against the coil set exactly as
        # they are against the baseline set. Identity when --coil_edges is off.
        edge_index = _CT.apply_coil_edges(edge_index, B, N, n_folded, device, CFG)
        edge_pairs = torch.unique(edge_index.t(), dim=0)
        edge_index = edge_pairs.t().contiguous()
        edge_attr = self.get_edge_attr(edge_index, ca_coords, B, N, device)
        return edge_index, edge_attr

    def get_edge_attr(self, edge_index, ca_coords, B, N, device):
        src = edge_index[0]
        dst = edge_index[1]
        batch_idx = src // N
        local_src = src % N
        local_dst = dst % N
        if ca_coords is not None:
            delta = ca_coords[batch_idx, local_src] - ca_coords[batch_idx, local_dst]
            dist = torch.norm(delta, dim=-1)
        else:
            dist = torch.zeros(src.shape[0], device=device)

        centers = torch.linspace(0.0, float(self.edge_cutoff), self.edge_rbf_dim, device=device)
        width = max(float(self.edge_cutoff) / max(self.edge_rbf_dim - 1, 1), 1e-6)
        rbf = torch.exp(-((dist.unsqueeze(-1) - centers) / width) ** 2)
        seq_sep = torch.abs(local_src - local_dst).float()
        seq_sep_norm = (seq_sep / max(N - 1, 1)).unsqueeze(-1)
        bonded = (seq_sep == 1).float().unsqueeze(-1)
        return torch.cat([rbf, seq_sep_norm, bonded], dim=-1)


def build_energy_model(
    model_arch=None,
    layers=None,
    gaussian_coef=None,
    dropout_rate=None,
    light_attention=None,
    emb_projection=None,
    gat_cutoff=None,
):
    """Build a DeepPEF energy model while keeping train/eval scripts architecture-agnostic."""
    model_arch = model_arch or CFG.model_arch
    layers = CFG.num_layers if layers is None else layers
    gaussian_coef = CFG.gaussian_coef if gaussian_coef is None else gaussian_coef
    dropout_rate = CFG.dropout_rate if dropout_rate is None else dropout_rate
    light_attention = CFG.light_attention if light_attention is None else light_attention
    emb_projection = CFG.emb_projection if emb_projection is None else emb_projection
    gat_cutoff = CFG.gat_cutoff if gat_cutoff is None else gat_cutoff

    if model_arch == "pem":
        return PEM(
            layers=layers,
            gaussian_coef=gaussian_coef,
            dropout_rate=dropout_rate,
            light_attention=light_attention,
            emb_projection=emb_projection,
            gat_cutoff=gat_cutoff,
        )
    if model_arch == "graph_transformer":
        return PEMGraphTransformer(
            layers=CFG.gt_layers,
            gaussian_coef=gaussian_coef,
            dropout_rate=dropout_rate,
            light_attention=light_attention,
            emb_projection=emb_projection,
            gt_hidden_dim=CFG.gt_hidden_dim,
            gt_heads=CFG.gt_heads,
            gt_edge_cutoff=CFG.gt_edge_cutoff,
            gt_edge_rbf_dim=CFG.gt_edge_rbf_dim,
        )
    raise ValueError(f"Unsupported model_arch: {model_arch}")
    
class PEMSM(torch.nn.Module):
  """Score matching Protein energy model"""
  
  def __init__(self, dim_in, dim_h, dim_out, layers, gaussian_coef,heads = 8):
    super().__init__()
    self.graph_model_gcn = [GCN(dim_in, dim_h, dim_out) for i in range(layers)]
    self.graph_model_gat = [GAT(dim_in, dim_h, dim_out) for i in range(layers)]

    self.gaussian_coef = gaussian_coef
    self.GAT_layers = torch.nn.ModuleList(self.graph_model_gat)
    self.GCN_layers = torch.nn.ModuleList(self.graph_model_gcn)
    # First fully connected layer
    self.fcs1 = nn.Linear(dim_in, 512)
    self.fcs2 = nn.Linear(512, dim_in)
    self.bn1  = nn.BatchNorm1d(dim_in)
    self.bn2  = nn.BatchNorm1d(dim_in)
    # First fully connected layer
    self.fc1 = nn.Linear(dim_in, 512)
    # Second fully connected layer that outputs our 10 labels
    self.fc2 = nn.Linear(512, 1)
  
      
  def forward(self,x,f_type = 'Default'):
      """
        Forward function
      Args:
          x_decoy (tensor): decoy coordinates [n_nodes, num_atoms=4, 3]
          emb_decoy (tensor): decoy embedding [n_nodes, emb_size]
          mask_decoy (tensor): decoy mask [n_nodes, 1]
          x_native (tensor): narive coordinates [n_nodes, num_atoms=4, 3]
          emb_native (tensor): native embedding [n_nodes, emb_size]
          mask_native (tensor): native mask [n_nodes, 1]
          edge_index (tensor): edge index [2, n_edges]
          f_type (str, optional): 'A_inference' or 'defualt', if 'A_inferece' return each amino acid energy . Defaults to 'Default'.

      Returns:
        if f_type == 'A_inference':
            energy: native and decoy energy for each amino acid
        if f_type == 'Default':
          energy: native and decoy energy
      """
      edge_index_gcn,edge_index_gat = self.get_edge_index(x)
      x = self.forward_x(x,edge_index_gcn,edge_index_gat)


    #   x_native  = self.get_graph(x_native, emb_native,mask_native)
    #   edge_index_gcn,edge_index_gat = self.get_edge_index(x_native)
    #   x_native = self.forward_x(x_native,edge_index_gcn,edge_index_gat)

      
      if (f_type == 'Default'):
        return self.get_energy(x)
      elif(f_type == 'A_inference'): # return the energy reference to each amino acid
        return x
        
  def forward_x(self,x,edge_index_gcn,edge_index_gat):
        """forward function for the graph model"""
        identity = x # identity for the residual connection
        x = x
        x = self.fcs1(x) # N,36->N,64
        x = F.relu(x)
        x = self.fcs2(x) # N,64->N,36
        x = self.bn1(x)
        for gcn_layer in self.GCN_layers:
            h1,x = gcn_layer(x, edge_index_gcn) 
            x = x + identity
        for gat_layer in self.GAT_layers:
            h1,x = gat_layer(x, edge_index_gat) 
            x = x + identity

        x = self.bn2(x)
        x  = self.fc1(x) # N,36->N,64
        x = F.relu(x)
        x = self.fc2(x) # N,64->N,1
        return x

  
  
  def get_energy(self,Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [n_nodes , embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sum(Fh ** 2, dim=(0, 1))
        return E
  
  def get_edge_index(self,x):
        seq_len = x.shape[0]
        combinations = torch.combinations(torch.arange(seq_len))
        edge_index_gat = combinations[combinations[:, 0] != combinations[:, 1]]
        edge_index_gat = edge_index_gat.t().contiguous().to(CFG.device)

        edge_index_gcn = torch.tensor([[i,i+1] for i in range(seq_len-1)]).t().contiguous().to(CFG.device)
        
        return edge_index_gcn,edge_index_gat
   
     
class GAT(torch.nn.Module):
  
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8, dropout_rate=0.2, edge_dim=None):
    super().__init__()
    # W7: edge_dim is None on the DEFAULT path, and these two constructor calls are then
    # character-for-character the historical ones -- same args, same RNG draw order, so
    # the OFF build is parameter-identical. Only edge_dim not None adds lin_edge.
    self.edge_dim = edge_dim
    if edge_dim is None:
      self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
      self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    else:
      self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, edge_dim=edge_dim)
      self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=edge_dim)
    # self.bn  = BatchNorm(dim_out)
    self.inst_norm = Normalization_layer(dim_out,affine=True)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x, edge_index, B, N, edge_attr=None):
    h=x
    h = self.dropout(x)
    # W7: with edge_dim None this is the historical two-argument call, unchanged.
    if self.edge_dim is None or edge_attr is None:
      h = self.gat1(h, edge_index)
      h = F.elu(h)
      h = self.gat2(h, edge_index)
    else:
      h = self.gat1(h, edge_index, edge_attr)
      h = F.elu(h)
      h = self.gat2(h, edge_index, edge_attr)
    # swap axis to use insrance norm
    h = h.reshape(B,N,-1)
    h = self.inst_norm(h)
    h = h.reshape(B*N,-1)
    
    return h, F.log_softmax(h, dim=1)


class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out, dropout_rate=0.2):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    self.inst_norm = Normalization_layer(dim_out,affine=True)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x, edge_index, B, N):
    h=x
    h = self.dropout(x)
    h = self.gcn1(h, edge_index)
    h = torch.relu(h)
    h = self.gcn2(h, edge_index)
    # swap axis to use insrance norm
    h = h.reshape(B,N,-1)
    h = self.inst_norm(h)
    h = h.reshape(B*N,-1)
    return h, F.log_softmax(h, dim=1)

class Normalization_layer(torch.nn.Module):
    """Normalization layer"""
    def __init__(self, dim_in,affine):
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(dim_in,affine=affine)
        # self.layer_norm = nn.LayerNorm(dim_in, elementwise_affine=affine)
    def forward(self, x):
        """forward function for the graph model
        Args:
            x (tensor): [batch_size ,n_nodes, dim_in]
        """
        # swap axis to use insrance norm
        x = x.transpose(1,2)
        x = self.inst_norm(x)
        x = x.transpose(1,2)
        
        # use layer norm
        # x = self.layer_norm(x)
        return x
    
class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                                padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                                padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o1 = o * self.softmax(attention)
        return o1
