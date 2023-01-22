"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class params():
    def __init__(self,embedding_size,layers,filters,cord_size,h):
        self.embedding_size = embedding_size
        self.layers = layers
        self.filters = filters
        self.cord_size = cord_size
        self.h = h

class ProteinEnergyNet(nn.Module):
    """
    The neural network.
    """

    def __init__(self,params,name='ProteinEnergyNet'):
        """
        In the constructor we instantiate GNN layers and assign them as member variables.
        inputs: 
            params: embedding_size, layers,cord_size,h 
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
        self.bonded = 1 
        # dirivative error
        self.h = params.h
        
        # corrdinate embedding paraameters
        self.KcoordsIn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3,self.cord_size))) # 3 for x,y,z
        self.KcoordsOut = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.cord_size,1))) 
        
        # GNN layers  - each layes contains the params for matrix multiplication(TODO: what is the benefit in convolution)
        self.Kbond_layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_layers,self.n_filters,
                                                                             self.n_atom_dist,self.emmbeding_size))) 
        self.Knonbond_layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_layers,self.n_filters,
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
            Energy [batch_size] tensor.
        """
        # Calculate energy for decoy and native
        E_xd = self.forward_x(X_decoy,emmbeidng)
        E_xn = self.forward_x(X_native,emmbeidng)
        
        return E_xd,E_xn

    def forward_x(self,X,emmbeidng):
        """
        Recives a single protein and calculate the energy
        Args:
            X (torch.tensor): Batch of proteins [batch_size,n_nodes ,num_atoms=4,coordination=3]
            emmbeidng (_type_): Batch of proteins [batch_size,n_nodes, embedding_size]

        Returns:
            E torch.tensor : Batch of proteins energy [batch_size]
        """
        B,N_residu,N_atoms,N_cords = X.shape
        Xembed = self.embed_cords(X)                                      # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        Fh,A_G = self.get_Fh0(Xembed,emmbeidng,self.h)                    # [batch_size, n_nodes ,atom_dist+embedding_size]
        # Start GNN layers loop:
        for layer in range(self.num_layers):
            # calculate avrege and gradient of each neigbor
            Ki = self.Knonbond_layers[layer]
            Ki_hat = self.Kbond_layers[layer]
            # Generate Fhb for bonded atoms
            Fhb = torch.zeros(B,N_residu,N_atoms,self.emmbeding_size+N_atoms**2)
            for i in range(self.bonded,Fh.shape[1],self.bonded):
                Fhb[:,(i-self.bonded):i,:]= self.layer_operation(Ki_hat,A_G[:,(i-self.bonded):i,:],
                                                                 Fh[:,(i-self.bonded):i,:])
            # Generate Fhub for noneboned atoms
            Fhub = self.layer_operation(Ki,A_G,Fh)
            # Update Feature vector for each node
            Fh = Fh-self.alpha*Fhub - self.alpha*Fhb
        # Calculate energy
        E = self.get_energy(Fh)
        
        return E
    
    def get_energy(Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [batch_size, n_nodes ,num_atoms=4, embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sqrt(torch.sum(Fh**2,dim=(1,2)))
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
        X_centered = X_decoy-X_decoy.mean(dim=-1, keepdim=True)
        X = torch.matmul(X_centered**2, self.KcoordsIn) #[batch_size, n_nodes ,num_atoms=4,new_cords_size]
        X = F.relu(X)
        X = torch.matmul(X, self.KcoordsOut)            #[batch_size, n_nodes ,num_atoms=4,new_cords_size]  
        return X * X_centered
   
    def layer_operation(self,Ki,A_G,Fh):
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
        Q = torch.matmul(A_G.reshape(B,atom_dist,N_residu),nodeE.reshape(B,N_residu,-1)) #[batch_size,atom_dist,embedding_size+n_nodes]
        Q = F.conv1d(Q, Ki)
        Q = F.instance_norm(Q)
        Q = F.leaky_relu(Q, negative_slope=0.2)
        Q = F.conv_transpose1d(Q, Ki)
        Q = torch.matmul(A_G.reshape(B,N_residu,atom_dist),Q)
        
        return Q
        
    
    def get_Fh0(self,Xd,FS,h):
        """
        Return the node features
        Args:
            Xd (tensor):X decoy [batch_size, n_nodes ,num_atoms=4,new_cords_size]
            FS (_type_): node features [batch_size,n_nodes, embedding_size]
            h (_type_): derovative step

        Returns:
            tuple of tensors : ([batch_size,n_nodes, embedding_size+atom_dist], [batch_size,n_nodes, n_atoms])
        """
        B,N_residu,N_atoms,coords_size = Xd.shape
        D = self.get_dist_matrix(Xd)                                             # [batch_size, n_nodes,n_nodes, atom_dist=16]
        # Get the derivative of the distance matrix
        G = (D[:,:,:,:-1]-D[:,:,:,1:])/(h)                                   # [batch_size, n_nodes,n_nodes, atom_dist_grad=15]
        # Get the average of the distance matrix
        A = 0.5*D.sum(dim=3,keepdim=True)                                   # [batch_size, n_nodes,n_nodes, atom_dist=16, 1]
        # First node features
        A_G = torch.cat((A,G),dim=3)                                        # [batch_size, n_nodes, n_nodes, atom_dist=16, 1]
        FD =  torch.matmul(A_G.reshape(B,N_residu,N_atoms**2,N_residu),D)   # [batch_size,n_nodes,atom_dist, atom_dist=16]
        FD = FD.sum(dim=3)                                                  # [batch_size,n_nodes, atoms_dist=16]
        FD = F.normalize(FD, p=2, dim=2)                                    # [batch_size,n_nodes, atoms_dist=16]
        Fh = torch.cat((FD,FS),dim=2)                                       # [batch_size,n_nodes, embedding_size+atoms_dist=16]
        return Fh,A_G
        
    def get_dist_matrix(self,Xd):
        """
        Return the node distence matrix
        Args:
            Xd (tensor):X embeded [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        Returns:
            tensor : [batch_size, n_nodes,n_nodes ,atom_dist=16] tensor
        """
        B,N_residu,N_atoms,coords_size = Xd.shape
        Xd = Xd.reshape(B,N_residu*N_atoms,coords_size)
        D = torch.cdist(Xd,Xd,p=2).reshape(B,N_residu,N_residu,N_atoms**2)      # [batch_size, n_nodes,n_nodes, atom_dist=16]
        return D