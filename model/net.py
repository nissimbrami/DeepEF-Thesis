"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.alpha = 0.0001
        self.bonded = 1
        # dirivative error
        self.h = params.h

        # corrdinate embedding paraameters
        self.KcoordsIn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3, self.cord_size)))  # 3 for x,y,z
        self.KcoordsOut = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.cord_size, 1)))

        # batch normalization
        self.epsilon = 1e-5
        self.momentum = 0.1
        self.gamma = nn.Parameter(torch.ones(self.emmbeding_size + self.n_atom_dist))
        self.beta = nn.Parameter(torch.zeros(self.emmbeding_size + self.n_atom_dist))
        self.register_buffer('running_mean', torch.zeros(self.emmbeding_size + self.n_atom_dist))
        self.register_buffer('running_var', torch.ones(self.emmbeding_size + self.n_atom_dist))

        # GNN layers  - each layes contains the params for matrix multiplication(TODO: what is the benefit in convolution)
        self.Kbond_layers = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_layers, self.n_filters,
                                                                            self.n_atom_dist + self.emmbeding_size,
                                                                            self.bonded)))
        self.Knonbond_layers = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_layers, self.n_filters,
                                                                               self.n_atom_dist + self.emmbeding_size,
                                                                               5)))

    def forward(self, X_decoy, X_native, embedding):
        """
        This is where we define the network's forward pass, i.e. how the network maps inputs to outputs.
        The forward pass wiill recive the input data as a tensor.
        Inputs:
            X_decoy: a [batch_size,n_nodes ,num_atoms=4,coordination=3] tensor
            X_native: a a [batch_size,n_nodes ,num_atoms=4,coordination=3] tensor
            emmbeidng: a [batch_size,n_nodes, embedding_size] tensor
        Since every node is connected to all other nodes there are no need for ajacency matrix.
        Returns:
            Energy [batch_size] tensor.
        """
        # Add f(x+h),f(x-h) to the input
        # X_decoyh = X_decoy + self.h
        # X_decoyl = X_decoy - self.h
        # X_decoy = torch.cat((X_decoy,X_decoyh,X_decoyl),dim=0) # [batch_size*3,n_nodes ,num_atoms=4,coordination=3]
        # X_nativeh = X_native + self.h
        # X_natively = X_native - self.h
        # X_native = torch.cat((X_native,X_nativeh,X_natively),dim=0)
        embedding = embedding.repeat(3, 1, 1)

        # Calculate energy for decoy and native
        E_xd = self.forward_x(X_decoy, embedding)
        E_xn = self.forward_x(X_native, embedding)
        # Concatenate the energy of the decoy and native
        E_xd = E_xd.unsqueeze(1)
        E_xn = E_xn.unsqueeze(1)
        return torch.cat((E_xd, E_xn), dim=1)

    def forward_x(self, X, embedding):
        """
        Recives a single protein and calculate the energy
        Args:
            X (torch.tensor): Batch of proteins [batch_size,n_nodes ,num_atoms=4,coordination=3]
            embedding (_type_): Batch of proteins [batch_size,n_nodes, embedding_size]

        Returns:
            E torch.tensor : Batch of proteins energy [batch_size]
        """
        B, N_residu, N_atoms, N_cords = X.shape
        # Xembed = self.embed_cords(X)                                      # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        Xembed = X
        Fh, A, G = self.get_Fh0(Xembed, embedding, self.h)  # [batch_size, n_nodes ,atom_dist+embedding_size]
        B = Fh.shape[0]
        # Start GNN layers loop:
        for layer in range(self.num_layers):
            Fh = self.normalize_graph(Fh)  # [batch_size, n_nodes ,atom_dist+embedding_size]
            # calculate avrege and gradient of each neigbor
            Ki = self.Knonbond_layers[layer]
            Ki_hat = self.Kbond_layers[layer]
            # Get new Avrege and gradient of each node
            A, G = self.get_avg_mat(Fh), self.get_Grad_mat(Fh)
            # Generate Fhb for bonded atoms
            Fhb = torch.zeros(B, N_residu, self.emmbeding_size + N_atoms ** 2, device=self.device)
            for i in range(self.bonded, Fh.shape[1], self.bonded):
                Fhb[:, (i - self.bonded):i, :] = self.layer_operation(Ki_hat,
                                                                      A[:, (i - self.bonded):i, (i - self.bonded):i],
                                                                      G[:, (i - self.bonded):i, (i - self.bonded):i],
                                                                      Fh[:, (i - self.bonded):i, :])
            # Generate Fhub for noneboned atoms
            Fhub = self.layer_operation(Ki, A, G, Fh)
            # Update Feature vector for each node
            Fh = Fh - self.alpha * Fhub - self.alpha * Fhb
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
            item: a [batch_size, n_nodes ,num_atoms=4,coordination=3] tensor
        Returns:
            X: a [batch_size, n_nodes ,num_atoms=4,new_cords_size, embedding_size] tensor
        
        3.1 equation from the research paper
        """
        X_centered = X_decoy - X_decoy.mean(dim=-1, keepdim=True)
        X = torch.matmul(X_centered ** 2, self.KcoordsIn)  # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        X = F.relu(X)
        X = torch.matmul(X, self.KcoordsOut)  # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        return X * X_centered

    def layer_operation(self, Ki, A, G, Fh):
        """
        Return the node features
        Args:
            K (tensor): weight matrix [n_filters,param1, param2]
            A (tensor): [batch_size, n_nodes, n_nodes]
            G (tensor): [batch_size, n_nodes, n_nodes]
            Fh (tensor): [batch_size,n_nodes, embedding_size+n_nodes]

        Returns:
            tensor : [batch_size,n_nodes, embedding_size+n_nodes]
        """
        B, N_residu, _ = A.shape
        nodeE = Fh
        Q = torch.matmul(A, nodeE) + torch.matmul(G, nodeE)  # [batch_size,n_nodes,embedding_size+atom_dist]
        # Change shape to fit the conv1d
        Q = Q.reshape(B, -1, N_residu)  # [batch_size,embedding_size+atom_dist,n_nodes]
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
        # D = D.reshape(B,N_residu,N_residu*N_atoms**2)                       # [batch_size, n_nodes,n_nodes*atom_dist=256]
        # Sum for each atom 16 distances
        D = D.sum(dim=2)  # [batch_size, n_nodes,atom_dist=16]
        # Get the h+1 and h-1 distance matrix
        Dr = D + D.mean() * h
        Dl = D - D.mean() * h
        D = torch.cat((D, Dr, Dl), dim=0)  # [batch_size*3, n_nodes,atom_dist=16*3]
        # Get the derivative of the distance matrix
        G = self.get_Grad_mat(D)  # [batch_size, n_nodes,n_nodes]
        # Get the average of the distance matrix
        A = self.get_avg_mat(D)  # [batch_size, n_nodes, n_nodes]
        # First node features
        FD = torch.matmul(A, D) + torch.matmul(G, D)  # [batch_size,n_nodes, atom_dist=16]
        # FD = F.normalize(FD, p=2, dim=2)                                    # [batch_size,n_nodes, atoms_dist=16]   
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
        B, N_residu, N_atoms, coords_size = Xd.shape
        Xd = Xd.reshape(B, N_residu * N_atoms, coords_size)
        D = torch.cdist(Xd, Xd, p=2).reshape(B, N_residu, N_residu,
                                             N_atoms ** 2)  # [batch_size, n_nodes,n_nodes, atom_dist=16]
        return D

    def get_avg_mat(self, Fh):
        """
        Return the node distance matrix between all nodes

        Args:
            Fh (tensor): tensor of node features [batch_size,n_nodes, embedding_size+n_nodes*atom_dist]
            
        output:
            AVG_MAT (tensor) : [batch_size,n_nodes, n_nodes] tensor
        """
        B, n_residue, _ = Fh.shape
        # Calculate the pairwise average between each node in the tensor
        pairwise_avg = (Fh.unsqueeze(axis=2) + Fh.unsqueeze(axis=1)) / 2
        return torch.sum(pairwise_avg, axis=-1)

    def get_Grad_mat(self, Fh):
        """
        Return the node distance matrix between all nodes

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

    def normalize_graph(self, x):
        """
        Preform batch normalization on the graph

        Args:
            x (tensor): B,N_residu,embedding_size+atom_dist tensor
        """
        # Compute the mean and variance of the node features for each graph
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_var = torch.var(x, dim=1, keepdim=True)

        # Normalize the node features
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.epsilon)

        # Weight and bias the normalized features
        x_out = self.gamma * x_norm + self.beta

        # Update the running mean and variance
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.squeeze()
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.squeeze()

        return x_out
