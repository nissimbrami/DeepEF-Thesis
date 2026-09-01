"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, BatchNorm
# v5: import the SAME CFG object that pnas_train_v5 mutates, so emb_input_dim
# (e.g. 1536 for dual_esmif) propagates to the model's feature-slice indices.
from model.model_cfg_v5 import CFG
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


class PEM(torch.nn.Module):
    """Protein energy model"""

    def __init__(self, layers, gaussian_coef, dropout_rate=0.2, light_attention=False,
                 emb_projection="none", gat_cutoff=None, serial_fusion=False,
                 serial_fusion_dim=64, use_learned_aa=False, aa_emb_dim=64,
                 length_norm=False):
        super().__init__()

        # v5: length/mass-balance normalization. When True, get_energy divides the summed
        # feature-energy by the number of residues, making the energy INTENSIVE (per-residue)
        # instead of EXTENSIVE (a raw sum that grows with protein length). This removes the
        # length bias in dG. See DeepPEF_v5/docs/V5_PLAN.md item 1.
        self.length_norm = length_norm

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

        # Serial fusion: project PLM embeddings INTO the GNN input
        self.serial_fusion = serial_fusion
        self.serial_fusion_dim = serial_fusion_dim
        if serial_fusion:
            self.serial_projector = MLPProjection(
                input_dim=CFG.emb_input_dim, hidden_dim=serial_fusion_dim * 2,
                output_dim=serial_fusion_dim, dropout=dropout_rate)

        # Learned amino acid embeddings (replaces 20-dim one-hot)
        self.use_learned_aa = use_learned_aa
        self.aa_emb_dim = aa_emb_dim
        if use_learned_aa:
            self.aa_embedding = nn.Embedding(20, aa_emb_dim)
        aa_dim = aa_emb_dim if use_learned_aa else 20

        # Dimensions depend on whether embeddings are projected into GNN
        # When projected: emb_proj_dim added to GNN input, not concatenated after
        proj_extra = emb_proj_dim if self.emb_projector is not None else 0
        serial_extra = serial_fusion_dim if serial_fusion else 0
        post_gnn_emb = 0 if self.emb_projector is not None else CFG.emb_input_dim

        # Lever C: burial scalar. When CFG.use_burial, a per-residue burial feature (width b) is
        # inserted BEFORE emb in the graph layout and fed into BOTH GNN towers, so every GNN /
        # fc input width grows by b. b=0 reproduces the baseline widths exactly.
        self.burial_dim = int(getattr(CFG, 'burial_dim', 0)) if getattr(CFG, 'use_burial', False) else 0
        b = self.burial_dim

        # Lever G: mutation-delta node feature (emb_mut - emb_wt). When CFG.mutation_delta, a
        # width-md block is inserted between emb and one_hot in the graph layout and fed into BOTH
        # GNN towers so the model sees WHAT CHANGED, not just the whole mutant embedding
        # (JanusDDG-style). md=0 reproduces the baseline widths exactly.
        self.mutation_delta_dim = int(getattr(CFG, 'mutation_delta_dim', 0)) if getattr(CFG, 'mutation_delta', False) else 0
        md = self.mutation_delta_dim

        # GNN internal dimension (what the graph layers operate on)
        # Base: 16 (dist) + aa_dim = 36 (default) or 16+aa_emb_dim (learned)
        gnn_internal_dim = 16 + aa_dim + proj_extra + serial_extra + b + md
        gcn_dim_in = gnn_internal_dim
        gcn_dim_h = 64
        gcn_dim_out = gnn_internal_dim
        self.graph_model_gcn = [GCN(gcn_dim_in, gcn_dim_h, gcn_dim_out, dropout_rate) for i in range(layers)]
        gat_dim_in = gnn_internal_dim
        gat_dim_h = 64
        gat_dim_out = gnn_internal_dim
        # Lever F: edge features. When CFG.use_edge_features, GATv2 consumes a 41-dim edge_attr.
        self.use_edge_features = bool(getattr(CFG, 'use_edge_features', False))
        edge_dim = 41 if self.use_edge_features else None
        self.graph_model_gat = [GAT(gat_dim_in, gat_dim_h, gat_dim_out, 8, dropout_rate, edge_dim=edge_dim) for i in range(layers)]
        # Gaussian coefficient
        self.gaussian_coef = gaussian_coef
        # graph attention layers
        self.GAT_layers = torch.nn.ModuleList(self.graph_model_gat)
        self.GCN_layers = torch.nn.ModuleList(self.graph_model_gcn)
        # Fully connected layers - GCN (projects from raw features to GNN dim)
        # GCN raw input: dist(16) + bonded(16) + aa_dim + proj_extra + serial_extra + burial(b) + delta(md)
        gcn_fc_in = 16 + 16 + aa_dim + proj_extra + serial_extra + b + md
        self.fc1_gcn = nn.Linear(gcn_fc_in, 64)
        self.fc2_gcn = nn.Linear(64, gcn_dim_in)
        # Fully connected layers - GAT (projects from raw features to GNN dim)
        # GAT raw input: dist(16) + aa_dim + proj_extra + serial_extra + burial(b) + delta(md)
        gat_fc_in = 16 + aa_dim + proj_extra + serial_extra + b + md
        self.fc1_gat = nn.Linear(gat_fc_in, 64)
        self.fc2_gat = nn.Linear(64, gat_dim_in)
        # normalization layers
        # GCN and GAT share the same internal dim, so use one norm layer (backward-compatible)
        self.inst_norm1 = Normalization_layer(gnn_internal_dim, affine=True)
        self.inst_norm2 = Normalization_layer(gcn_dim_out + gat_dim_out, affine=True)
        # Fc layers for the final output
        fc_in_dim = gcn_dim_out + gat_dim_out + post_gnn_emb
        self.fc1 = nn.Linear(fc_in_dim, 128)
        # Lever A: energy decomposition. fc2 emits K per-residue energy TERMS (K=1 == baseline
        # scalar). get_energy sums the K terms per residue, then (optionally length_norm) sums
        # over residues. We store the last per-term energies for interpretability.
        self.energy_terms = max(int(getattr(CFG, 'energy_terms', 1)), 1)
        self.fc2 = nn.Linear(128, self.energy_terms)
        self.last_term_energies = None  # [B, K] per-term totals, filled in get_energy

        # GNN-SM output head: per-residue amino acid scores [L, 20]
        self.fc2_sm = nn.Linear(128, 20)

        # Lever E: denoising auxiliary head. When CFG.denoise_weight>0, an MLP off the 128-dim
        # pre-energy features predicts the injected per-atom coordinate noise (4 atoms * 3 = 12
        # values per residue). Head absent (None) reproduces the baseline exactly.
        self.denoise_weight = float(getattr(CFG, 'denoise_weight', 0.0))
        if self.denoise_weight > 0:
            self.denoise_head = nn.Sequential(
                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4 * 3))
        else:
            self.denoise_head = None
        self.last_denoise_pred = None  # [B, N, 4, 3] cached in forward when denoise head is on

        # Lever F: extended GCN connectivity span (S=1 -> baseline i,i+1 edges only).
        self.gcn_span = max(int(getattr(CFG, 'gcn_span', 1)), 1)

        # energy epsilon
        self.energy_epsilon = 1

        # embedding indexes — dynamically computed in forward
        self.aa_dim = aa_dim
        self.one_hot_index = -20  # legacy: used only without learned_aa
        self.bonded_index = 48
        self.non_bonded_index = 16
        # Lever G: mutation-delta block (width md, from the ctor above) sits between emb and
        # one_hot, so both the emb and delta slices are anchored from the right. md=0 reproduces
        # the baseline indices exactly.
        self.delta_index = -(20 + md)               # start of the delta block (from the right)
        self.llm_index = -(CFG.emb_input_dim + 20 + md)  # emb block sits left of the delta block
        # Lever C: burial occupies [dist_dim*3 : dist_dim*3 + burial_dim] i.e. right after the
        # 48 dist+bonded columns (dist_dim*3 = 16*3 = 48). one_hot/emb stay at from-the-right.
        self.burial_start = CFG.dist_dim * 3

        # edge index cache
        self._edge_cache_key = None
        self._edge_cache = None

        # GAT distance cutoff (Angstroms); None = fully connected
        self.gat_cutoff = gat_cutoff

        # light attention machanism
        self.light_attention = light_attention
        if self.light_attention:
            self.LA = LightAttention(embeddings_dim=fc_in_dim)
        
    
    def forward(self,x,f_type = 'Default', ca_coords=None):
        """
                Forward function
             Args:
            x (tensor): [batch, n_nodes, bonded_features+non_bonded_features+LLM_features]
            f_type (str, optional): 'A_inference' or 'defualt', if 'A_inferece' return each amino acid energy . Defaults to 'Default'.
            ca_coords (tensor, optional): [batch, n_nodes, 3] CA atom coordinates for distance-based GAT edges.

        Returns:
            if f_type == 'A_inference':
                energy: native and decoy energy for each amino acid
            if f_type == 'Default':
            energy: native and decoy energy
        """
        # Get the edge index
        edge_index_gcn,edge_index_gat = self.get_edge_index(x, ca_coords=ca_coords)
        # reshape x to [batch_size*n_nodes,features]
        B, N, _ = x.shape
        x = x.reshape(B * N,-1)
        # split features: layout is
        #   [dist:16 | bonded:32 | (burial:b) | emb:E | (delta:md) | one_hot:20]
        # GCN uses only first 16 of bonded (bonded-to-previous), matching original behavior
        x_dist = x[:, :self.non_bonded_index]  # B*N, 16
        x_bonded = x[:, self.non_bonded_index:self.non_bonded_index * 2]  # B*N, 16
        x_emb_features = x[:, self.llm_index:self.delta_index]  # B*N, E (from the right, left of delta)
        x_onehot = x[:, self.one_hot_index:]  # B*N, 20 (always last)
        # Lever G: mutation-delta slice sits between the emb block and one_hot.
        if self.mutation_delta_dim > 0:
            x_delta = x[:, self.delta_index:self.one_hot_index]  # B*N, md
            if x_delta.shape[1] != self.mutation_delta_dim:
                raise ValueError(
                    f"[Lever G] mutation-delta slice width {x_delta.shape[1]} != "
                    f"mutation_delta_dim {self.mutation_delta_dim}; node feature vector shape "
                    f"mismatch. Graph builder / model layout disagree.")
        else:
            x_delta = None
        # Lever C: burial slice sits between the 48 dist+bonded cols and the emb block.
        if self.burial_dim > 0:
            x_burial = x[:, self.burial_start:self.burial_start + self.burial_dim]  # B*N, b
            # Fail-fast: a short/misaligned feature vector would make this slice the wrong width
            # and only blow up later in a GNN matmul. Name the lever now.
            if x_burial.shape[1] != self.burial_dim:
                raise ValueError(
                    f"[Lever C] burial slice width {x_burial.shape[1]} != burial_dim "
                    f"{self.burial_dim}; node feature vector shorter than expected "
                    f"(burial_start={self.burial_start}). Graph builder / model layout mismatch.")
        else:
            x_burial = None

        # Learned AA embeddings: convert one-hot to learned embedding
        if self.use_learned_aa:
            aa_indices = x_onehot.argmax(dim=-1)  # B*N
            x_aa = self.aa_embedding(aa_indices)  # B*N, aa_emb_dim
        else:
            x_aa = x_onehot  # B*N, 20

        # Build GCN input: dist(16) + bonded(16) + aa (+ burial)
        x_gcn = torch.cat((x_dist, x_bonded, x_aa), dim=-1)  # B*N, 32+aa_dim
        # Build GAT input: dist(16) + aa (+ burial)
        x_gat = torch.cat((x_dist, x_aa), dim=-1)  # B*N, 16+aa_dim
        if x_burial is not None:
            x_gcn = torch.cat((x_gcn, x_burial), dim=-1)
            x_gat = torch.cat((x_gat, x_burial), dim=-1)

        # Lever G: feed the mutation-delta block into BOTH towers so message-passing sees the
        # change signal (emb_mut - emb_wt). Appended after burial; widths tracked in the ctor.
        if x_delta is not None:
            x_gcn = torch.cat((x_gcn, x_delta), dim=-1)
            x_gat = torch.cat((x_gat, x_delta), dim=-1)

        # Project embeddings and concatenate into GNN input, or keep for post-GNN concat
        if self.emb_projector is not None:
            x_proj = self.emb_projector(x_emb_features)  # B*N, proj_dim
            x_gcn = torch.cat((x_gcn, x_proj), dim=-1)
            x_gat = torch.cat((x_gat, x_proj), dim=-1)

        # Serial fusion: project PLM into GNN input for message-passing
        if self.serial_fusion:
            x_serial = self.serial_projector(x_emb_features)  # B*N, serial_fusion_dim
            x_gcn = torch.cat((x_gcn, x_serial), dim=-1)
            x_gat = torch.cat((x_gat, x_serial), dim=-1)

        # forward pass through the graph attention and convolution layers
        x1 = self.forward_gcn(x_gcn, edge_index_gcn, B, N)
        # Lever F: build 41-dim edge_attr [onehot_src(20)|onehot_dst(20)|dist(1)] for GAT edges.
        edge_attr = None
        if self.use_edge_features:
            edge_attr = self.build_edge_attr(edge_index_gat, x_onehot, ca_coords, B, N)
        x2 = self.forward_gat(x_gat, edge_index_gat, B, N, edge_attr=edge_attr)
        # concat features
        x = torch.cat((x1,x2),dim=-1)
        # reshape to use instance norm
        x = x.reshape(B, N,-1)
        x = self.inst_norm2(x)
        x = x.reshape(B * N,-1)
        # Add raw LLM features only when no projection (original behavior)
        if self.emb_projector is None:
            x = torch.cat((x,x_emb_features),dim=-1)
        # Light attention machanism
        if self.light_attention:
            x = x.reshape(B, N,-1)
            x = x.swapaxes(1,2)
            x = self.LA(x)
            x = x.swapaxes(1,2)
            x = x.reshape(B * N,-1)
        # fc layers
        x  = self.fc1(x)
        x = F.relu(x)

        # Lever E: predict injected coordinate noise from the 128-dim pre-energy features.
        if self.denoise_head is not None:
            dn = self.denoise_head(x)              # B*N, 12
            self.last_denoise_pred = dn.reshape(B, N, 4, 3)
        else:
            self.last_denoise_pred = None

        # Branch: subtract-mut mode returns [B, L, 20] scores
        if f_type == 'subtract_mut':
            x_sm = self.fc2_sm(x)  # B*N, 20
            x_sm = x_sm.reshape(B, N, 20)
            return x_sm

        x = self.fc2(x) # -> B*N,K  (K=energy_terms; K=1 == baseline)
        # reshape to [batch_size, n_nodes, K]
        x = x.reshape(B, N, self.energy_terms)
        # return energy
        if (f_type == 'Default'):
            return self.get_energy(x)
        elif(f_type == 'A_inference'): # return the per-residue (per-term) energy
            return x
        
    def forward_gat(self, x, edge_index_gat, B, N, edge_attr=None):
        """forward function for the graph model.
        Lever F: edge_attr (41-dim [onehot_src|onehot_dst|dist]) is passed through to each
        GATv2 layer when use_edge_features is on; None reproduces the baseline attention."""
        identity = x # identity for the residual connection
        x = self.fc1_gat(x) # N,gat_in->N,64
        x = F.relu(x)
        x = self.fc2_gat(x) # N,64->N,gat_in
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
        x = self.fc1_gcn(x) # N,gcn_in->N,64
        x = F.relu(x)
        x = self.fc2_gcn(x) # N,64->N,gcn_in
        # swap axis to use insrance norm
        x = x.reshape(B, N,-1)
        x = self.inst_norm1(x)
        x = x.reshape(B * N,-1)
        identity = x # identity for the residual connection
        for gcn_layer in self.GCN_layers:
            h1,z = gcn_layer(x, edge_index_gcn, B, N)
            x = h1 + identity
        return x

    def build_edge_attr(self, edge_index_gat, x_onehot, ca_coords, B, N):
        """Lever F: build a per-edge feature vector for the GAT edges.

        edge_attr = [ onehot_src(20) | onehot_dst(20) | ca_distance(1) ] -> 41 dims.
        Args:
            edge_index_gat: [2, E] flat (batch-offset) node indices.
            x_onehot: [B*N, 20] one-hot amino-acid codes (flattened batch).
            ca_coords: [B, N, 3] CA coordinates (used for the pairwise distance term).
            B, N: batch size and residues-per-graph.
        Returns:
            [E, 41] edge feature tensor on the same device as x_onehot.
        """
        src, dst = edge_index_gat[0], edge_index_gat[1]
        oh_src = x_onehot[src]  # [E, 20]
        oh_dst = x_onehot[dst]  # [E, 20]
        if ca_coords is not None:
            ca_flat = ca_coords.reshape(B * N, 3)
            dist = (ca_flat[src] - ca_flat[dst]).norm(dim=-1, keepdim=True)  # [E, 1]
        else:
            dist = torch.zeros(src.shape[0], 1, device=x_onehot.device, dtype=x_onehot.dtype)
        edge_attr = torch.cat([oh_src, oh_dst, dist], dim=-1)  # [E, 41]
        # Lever F fail-fast: GATv2Conv was built with edge_dim=41; a mismatch here would raise a
        # cryptic scatter error inside the conv. Surface it with the lever name instead.
        if edge_attr.shape[-1] != 41:
            raise ValueError(
                f"[Lever F] edge_attr width {edge_attr.shape[-1]} != 41 "
                f"([onehot_src(20)|onehot_dst(20)|dist(1)]). one_hot width must be 20.")
        return edge_attr  # [E, 41]

    def get_energy(self,Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [batch, n_nodes, K] tensor of per-residue energy TERMS (K=energy_terms).
        Returns:
            Energy [batch_size] tensor
        Lever A: sum the K per-residue terms into a per-residue energy, then sum over residues
                 (K=1 is identical to the historical single-scalar readout). Per-term totals are
                 cached in self.last_term_energies for interpretability.
        v5: when self.length_norm is True, divide by the number of residues so the energy is
        intensive (per-residue) rather than extensive (length-dependent sum).
        """
        # Per-term total over residues: [B, K] (kept for interpretability / diagnostics).
        self.last_term_energies = torch.sum(Fh, dim=1).detach()
        # Total energy = sum over residues AND terms (matches baseline sum(dim=(1,2)) for K=1).
        E = torch.sum(Fh, dim=(1, 2))
        if self.length_norm:
            n_res = Fh.shape[1]
            E = E / max(n_res, 1)
        # E = torch.log(torch.sum(Fh,dim=(1,2)) + self.energy_epsilon)
        return E
  
    def get_edge_index(self, x, ca_coords=None):
        """Return the edge index for the graph convolution and attention layers.
        The edge index of the gcn is a line from the amino acid to the next amino acid.
        The edge index of the gat uses a distance cutoff on CA atoms when ca_coords
        is provided and self.gat_cutoff is set; otherwise falls back to fully connected.
        Results are cached only for the fully-connected (shape-based) case."""
        B, N = x.shape[0], x.shape[1]

        # Cache only when edges depend solely on shape (no ca_coords)
        if ca_coords is None:
            key = (B, N)
            if self._edge_cache_key == key and self._edge_cache is not None:
                return self._edge_cache

        # GCN: sequential edges within each batch element.
        # Baseline (gcn_span=1): directed (i, i+1) edges only -> bit-exact.
        # Lever F: for span S>1, add offsets 1..S in BOTH directions.
        dev = x.device
        offsets = torch.arange(B, device=dev).unsqueeze(1) * N  # [B, 1]
        if self.gcn_span <= 1:
            local_gcn = torch.arange(N - 1, device=dev)
            gcn_src = (local_gcn.unsqueeze(0) + offsets).reshape(-1)
            gcn_dst = gcn_src + 1
            edge_index_gcn_all = torch.stack([gcn_src, gcn_dst])
        else:
            src_parts, dst_parts = [], []
            for s in range(1, self.gcn_span + 1):
                if N - s <= 0:
                    continue
                local = torch.arange(N - s, device=dev)
                base_src = (local.unsqueeze(0) + offsets).reshape(-1)
                base_dst = base_src + s
                # forward edge (i -> i+s)
                src_parts.append(base_src)
                dst_parts.append(base_dst)
                # backward edge (i+s -> i)
                src_parts.append(base_dst)
                dst_parts.append(base_src)
            gcn_src = torch.cat(src_parts)
            gcn_dst = torch.cat(dst_parts)
            edge_index_gcn_all = torch.stack([gcn_src, gcn_dst])

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

        if ca_coords is None:
            self._edge_cache_key = (B, N)
            self._edge_cache = (edge_index_gcn_all, edge_index_gat_all)

        return edge_index_gcn_all, edge_index_gat_all
    
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
    # Lever F: pass edge_dim so GATv2 attends over edge features [onehot_src|onehot_dst|dist].
    self.edge_dim = edge_dim
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, edge_dim=edge_dim)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=edge_dim)
    # self.bn  = BatchNorm(dim_out)
    self.inst_norm = Normalization_layer(dim_out,affine=True)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x, edge_index, B, N, edge_attr=None):
    h=x
    h = self.dropout(x)
    h = self.gat1(h, edge_index, edge_attr=edge_attr)
    h = F.elu(h)
    h = self.gat2(h, edge_index, edge_attr=edge_attr)
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