"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinEnergyNet(nn.Module):
    """
    The neural network.
    """

    def __init__(self,params,name='ProteinEnergyNet'):
        """
        In the constructor we instantiate GNN layers and assign them as member variables.
        inputs: 
            params: [embedding_size, num_heads, layers,cord_size] dictionary
            name: name of the network
        """
        super(ProteinEnergyNet, self).__init__()
        self.name = name
        self.num_layers = params.layers
        self.cord_size = params.cord_size
        # corrdinate embedding paraameters
        self.KcoordsIn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3,self.cord_size))) # 3 for x,y,z
        self.KcoordsOut = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.cord_size))) # 3 for x,y,z
        
        # GNN layers
        self.Kbond_layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_layers,self.cord_size))) # 3 for x,y,z
        self.Kbond_layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_layers,self.cord_size))) # 3 for x,y,z
        
        for i in range(self.num_layers):
            self.GNN_layers[i] = nn.MultiheadAttention(params.embedding_size, params.num_heads)
        

    def forward(self, X_decoy, X_native,emmbeidng):
        """
        This is where we define the network's forward pass, i.e. how the network maps inputs to outputs.
        The forward pass wiill recive the input data as a tensor.
        Inputs:
            X_decoy: a [batch_size,n_nodes ,num_atoms=4,coordination=3] tensor
            X_native: a a [batch_size,n_nodes ,num_atoms=4,coordination=3] tensor
            emmbeidng: a [batch_size,n_nodes, embedding_size] tensor
        Since every node is connected to all other nodes there are no need for ajacency matrix.
        Returns:
            Energy [batch_size] tensor
        """
        B,N_residu,N_atoms,N_cords = X_decoy.shape
        Xd = self.embed_cords(X_decoy,self.cord_size)                       # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        D = self.getDistMatrices(Xd.reshape(B,N_atoms,N_cords,N_residu))    # [batch_size, n_nodes, atom_dist=16, new_cords_size]    
        Fd = self.layer_operation(D,emmbeidng)                              # [batch_size,n_nodes, embedding_size]
        FS =  emmbeidng                                                     # [batch_size,n_nodes, embedding_size]


    def embed_cords(self, X_decoy):
        """
        Embeds the item into a vector representation.
        Inputs:
            item: a [batch_size, n_nodes ,num_atoms=4,coordination=3] tensor
        Returns:
            X: a [batch_size, n_nodes ,num_atoms=4,new_cords_size, embedding_size] tensor
        
        3.1 equation from the research paper
        """
        X_mean = X_decoy-X_decoy.mean(dim=-1, keepdim=True)
        X = torch.matmul(X_mean**2, self.KcoordsIn) #[batch_size, n_nodes ,num_atoms=4,new_cords_size]
        X = torch.nn.ReLU(X)
        X = torch.matmul(X, self.KcoordsOut) #[batch_size, n_nodes ,num_atoms=4,new_cords_size]  
        return X * X_mean
    
    def getDistMatrices(Coords):
        # Coords is assumed to be of shape [B, 4, 3, N]
        # Compute distance maps
        batchSize = Coords.shape[0]
        nnodes = Coords.shape[-1]
        I = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        J = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        DI = torch.sum(Coords[:, I, :, :] ** 2, dim=2, keepdim=True)
        DJ = torch.sum(Coords[:, J, :, :] ** 2, dim=2, keepdim=True)
        DIJ = DI + DJ.transpose(3, 2)

        XI = Coords[:, I, :, :]
        XJ = Coords[:, J, :, :]
        XIXJ = torch.bmm(XJ.reshape(batchSize * 16, 3, -1).transpose(1, 2), XI.reshape(batchSize * 16, 3, -1))
        XTX = DIJ - 2 * XIXJ.reshape(batchSize, 16, nnodes, nnodes)
        XTX = torch.relu(XTX)

        return XTX