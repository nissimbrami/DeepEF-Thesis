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
            params: [embedding_size, num_heads, layers,cord_size,h] dictionary
            name: name of the network
        """
        super(ProteinEnergyNet, self).__init__()
        self.name = name
        # GNN parameters
        self.num_layers = params.layers
        self.n_filters = params.filters
        self.cord_size = params.cord_size
        self.n_atom_dist = 16
        self.emmbeding_size = params.embedding_size
        self.alpha = 0.2
        self.bonded = 4
        # dirivative error
        self.h = params.h
        
        # corrdinate embedding paraameters
        self.KcoordsIn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3,self.cord_size))) # 3 for x,y,z
        self.KcoordsOut = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.cord_size))) # 3 for x,y,z
        
        # GNN layers  - each layes contains the params for matrix multiplication(TODO: what is the benefit in convolution)
        self.Kbond_layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_layers,self.n_filters,
                                                                             self.n_atom_dist,self.emmbeding_size))) 
        self.Kunbond_layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_layers,self.n_filters,
                                                                             self.n_atom_dist,self.emmbeding_size)))
        
       
        

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
        # Calculate energy for decoy
        E_xd = self.forward_x(X_decoy,emmbeidng)
        E_xn = self.forward_x(X_native,emmbeidng)
        
        return E_xd,E_xn

    def forward_x(self,X,emmbeidng):
        B,N_residu,N_atoms,N_cords = X.shape
        Xd = self.embed_cords(X,self.cord_size)                       # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        Fh,A_G = self.get_Fh0(Xd,emmbeidng,self.h)
        # Start GNN layers loop:
        for layer in self.num_layers:
            # calculate avrege and gradient of each neigbor
            Ki = self.Kunbond_layers[layer]
            Ki_hat = self.Kbond_layers[layer]
            # Generate Fhb for bonded atoms
            Fhb = torch.zeros(B,N_residu,N_atoms,self.emmbeding_size+N_residu)
            for i in range(self.bonded,Fh.shape[1],self.bonded):
                Fhb[:,(i-self.bonded):i,:,:]= self.layer_operation(Ki_hat,A_G[:,(i-self.bonded):i,:,:],
                                                                   Fh[:,(i-self.bonded):i,:,:])
            # Generate Fhub for unboned atoms
            Fhub = self.layer_operation(Ki,A_G,Fh)
            Fh = Fh-self.alpha*Fhub - self.alpha*Fhb
        # Calculate energy
        return self.get_energy(Fh)
    
    def get_energy(Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [batch_size, n_nodes ,num_atoms=4, embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sqrt(torch.sum(Fh**2,dim=(1,2,3)))
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
        X_mean = X_decoy-X_decoy.mean(dim=-1, keepdim=True)
        X = torch.matmul(X_mean**2, self.KcoordsIn) #[batch_size, n_nodes ,num_atoms=4,new_cords_size]
        X = torch.nn.ReLU(X)
        X = torch.matmul(X, self.KcoordsOut) #[batch_size, n_nodes ,num_atoms=4,new_cords_size]  
        return X * X_mean
   
    def layer_operation(Ki,A_G,Fh):
        """
        Return the node features
        Args:
            K (tensor): weight matrix [n_filters,param1, param2]
            A_G (tensor): [batch_size, n_nodes, atoms_dist=16]
            Fh (tensor): [batch_size,n_nodes, embedding_size+n_nodes]

        Returns:
            tensor : [batch_size,n_nodes, embedding_size+n_nodes]
        """
        B,N_residu,atom_dist = A_G.shape
        nodeE = Fh
        Q = torch.matmul(A_G.reshape(B,atom_dist,N_residu),nodeE.reshape(B,N_residu,-1))
        Q = F.conv1d(Q, Ki)
        Q = F.instance_norm(Q)
        Q = F.leaky_relu(Q, negative_slope=0.2)
        Q = F.conv_transpose1d(Q, Ki)
        Q = torch.matmul(A_G.reshape(B,N_residu,atom_dist),Q)
        
        return Q
        
    def getDistMatrices(Coords):
        # Coords is assumed to be of shape [B, 4, 3, N]
        # Compute distance maps and returns [batch_size, n_nodes, atom_dist=16, new_cords_size] tensor
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
    
    def get_Fh0(Xd,FS,h):
        """
        Return the node features
        Args:
            Xd (tensor):X decoy [batch_size, n_nodes ,num_atoms=4,new_cords_size]
            FS (_type_): node features [batch_size,n_nodes, embedding_size]
            h (_type_): derovative step

        Returns:
            tuple of tensors : ([batch_size,n_nodes, embedding_size+n_nodes], [batch_size,n_nodes, n_atoms])
        """
        B,N_residu,N_atoms,coords_size = Xd.shape
        D = torch.cdist(Xd,Xd,p=2).reshape(B,N_residu,N_atoms*N_atoms)      # [batch_size, n_nodes, atom_dist=16]
        # Get the derivative of the distance matrix
        G = torch.abs((D[:,:,:-1]-D[:,:,1:])/(h))                           # [batch_size, n_nodes, atom_dist=16, 1]
        # Get the average of the distance matrix
        A = 0.5*D.sum(dim=2,keepdim=True)                                   # [batch_size, n_nodes, atom_dist=16, 1]
        # First node features
        print(A.shape,G.shape,D.shape)
        A_G = torch.cat((A,G),dim=2)                                        # [batch_size, n_nodes, atom_dist=16, 1]
        FD =  torch.matmul(A_G,D.reshape(B,N_atoms**2,N_residu))            # [batch_size,n_nodes, n_nodes]
        Fh = torch.cat((FD,FS),dim=2)                                       # [batch_size,n_nodes, embedding_size+n_nodes]
        return Fh,A_G