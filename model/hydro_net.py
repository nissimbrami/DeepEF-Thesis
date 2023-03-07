"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
# import matplotlib.pyplot as plt

class params():
    def __init__(self,embedding_size,layers,filters,cord_size,h,device):
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

    def __init__(self,params,name='ProteinEnergyNet'):
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
        # corrdinate embedding paraameters
        sigma = 1+torch.zeros(3*self.n_atom_dist, self.n_atom_dist, 5 , 5)
        self.sigma = nn.Parameter(sigma)
        self.biasDistance = nn.Parameter(0.6*torch.ones(1, 3*self.n_atom_dist, 1, 1))
        self.KcoordsIn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3,self.cord_size))) # 3 for x,y,z
        self.KcoordsOut = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.cord_size,1))) 
        
        # GNN layers  - each layes contains the params for matrix multiplication(TODO: what is the benefit in convolution)
        self.Kbond_layers = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_layers,self.n_filters,
                                                                             3*self.n_atom_dist+ self.emmbeding_size,self.bonded))) 
        self.Knonbond_layers = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_layers,self.n_filters,
                                                                             3*self.n_atom_dist+ self.emmbeding_size,11)))
        
       
        

    def forward(self, X_decoy, X_native,embeiddng,emb_decoy):
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
        E_xd = self.forward_x(X_decoy,emb_decoy)
        E_xn = self.forward_x(X_native,embeiddng)
        # Concatenate the energy of the decoy and native
        E_xd = E_xd.unsqueeze(1)
        E_xn = E_xn.unsqueeze(1)
        return torch.cat((E_xd,E_xn),dim=1)

    def forward_x(self,X,embeiddng):
        """
        Recives a single protein and calculate the energy
        Args:
            X (torch.tensor): Batch of proteins [batch_size,n_nodes ,num_atoms=4,coordination=3]
            emmbeidng (_type_): Batch of proteins [batch_size,n_nodes, embedding_size]

        Returns:
            E torch.tensor : Batch of proteins energy [batch_size]
        """
        B,N_residu,N_atoms,N_cords = X.shape
        #Xembed = self.embed_cords(X)                                      # [batch_size, n_nodes ,num_atoms=4,new_cords_size]
        # X_centered = X-X.mean(dim=1, keepdim=True)
        # Xembed = X_centered
        Xembed = X
        Fh,A,G = self.get_Fh0(Xembed,embeiddng,self.h)                    # [batch_size, n_nodes ,atom_dist+embedding_size]
        B,N,d = Fh.shape
        #Start GNN layers loop:
        for layer in range(self.num_layers):
            # calculate avrege and gradient of each neigbor
            Ki = self.Knonbond_layers[layer]
            Ki_hat = self.Kbond_layers[layer]
            # Get new Avrege and gradient of each node
            A , G = self.get_AVG_mat(Fh), self.get_Grad_mat(Fh)
            # Generate Fhb for bonded atoms
            Fhb = torch.zeros(B,N_residu,self.emmbeding_size+3*N_atoms**2,device=self.device)
            for i in range(self.bonded,Fh.shape[1],self.bonded):
                Fhb[:,(i-self.bonded):i,:]= self.layer_operation(Ki_hat,A[:,(i-self.bonded):i,(i-self.bonded):i],G[:,(i-self.bonded):i,(i-self.bonded):i],
                                                     Fh[:,(i-self.bonded):i,:])
            # Generate Fhub for noneboned atoms
            Fhub = self.layer_operation(Ki,A,G,Fh)
            # Update Feature vector for each node
            Fh = Fh-self.alpha*Fhub - self.alpha*Fhb
            Fh = F.normalize(Fh, p=2, dim=1)         
        # Calculate energy
        E = self.get_energy(Fh)
        
        return E
    
    def get_energy(self,Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [batch_size, n_nodes , embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sum(Fh**2,dim=(1,2))
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
        X_centered = X_decoy-X_decoy.mean(dim=1, keepdim=True)
        X = torch.matmul(X_centered**2, self.KcoordsIn) #[batch_size, n_nodes ,num_atoms=4,new_cords_size]
        X = F.relu(X)
        X = torch.matmul(X, self.KcoordsOut)            #[batch_size, n_nodes ,num_atoms=4,new_cords_size]  
        return X * X_centered
   
    def layer_operation(self,Ki,A,G,Fh):
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
        B,N_residu,_ = A.shape
        nodeE = Fh
        Q = torch.matmul(A,nodeE) + torch.matmul(G,nodeE)  #[batch_size,n_nodes,d]
        # Change shape to fit the conv1d
        Q = Q.reshape(B,-1,N_residu)                       #[batch_size,d,n_nodes]
        Q = F.conv1d(Q, Ki)
        # Q = F.instance_norm(Q)
        Q = F.leaky_relu(Q, negative_slope=0.2)
        Q = F.conv_transpose1d(Q, Ki)
        Q = Q.reshape(B,N_residu,-1)                       #[batch_size, n_nodes, filters]
        Q = torch.matmul(A,Q) + torch.matmul(G,Q)   #[batch_size,n_nodes,embedding_size+n_nodes]
        return Q
        
    
    def get_Fh0(self,Xd,FS,h):
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
        B,N_residu,N_atoms,coords_size = Xd.shape
        D = self.get_dist_matrix(Xd)                                        # [batch_size, n_nodes,n_nodes, atom_dist=16]-> [batch_size,n_nodes*atom_dist, n_nodes]
        # Compute the gussian of the distance matrix
        D = torch.swapaxes(torch.swapaxes(D,3,2),2,1)#D.reshape(B,N_atoms**2,N_residu,N_residu)              # [batch_size, n_nodes*atom_dist,n_nodes]
        Z = F.conv2d(D, self.sigma.abs(), padding=self.sigma.shape[-1]//2)
        Z = F.normalize(Z, dim=[2,3])
        D = torch.relu(torch.exp(-1e1*Z) - self.biasDistance)
        # Sum for each atom 16 distances
        D = D.sum(dim=2)                                                   # [batch_size, n_nodes,atom_dist=16]
        D = F.normalize(D, p=2, dim=1)                                    # [batch_size,n_nodes, atoms_dist=16]
        D = torch.swapaxes(D,1,2)#D.reshape(B,N_residu,-1)
        # Get the derivative of the distance matrix
        G = self.get_Grad_mat(D)                                        # [batch_size, n_nodes,n_nodes]
        # Get the average of the distance matrix
        A = self.get_AVG_mat(D)                                             # [batch_size, n_nodes, n_nodes]
        # First node features
        FD =  torch.matmul(A,D) +torch.matmul(G,D)                          # [batch_size,n_nodes, atom_dist=16]
        #TODO: diffrences between atomes of the same node are small
        FD = F.normalize(FD, p=2, dim=1)                                    # [batch_size,n_nodes, atoms_dist=16]   
        Fh = torch.cat((FD,FS),dim=2)                                       # [batch_size,n_nodes, embedding_size+atoms_dist=16]
        return Fh,A,G
        
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
        D = torch.cdist(Xd,Xd,p=2)
        D = D.reshape(B,N_residu,N_atoms,N_residu,N_atoms)
        D = torch.swapaxes(D,2,3)
        D = D.reshape(B,N_residu,N_residu,N_atoms*N_atoms)
        return D
    
    def get_AVG_mat(self,Fh):
        """
        Return the node distence matrix between all nodes

        Args:
            Fh (tensor): tensor of node features [batch_size,n_nodes, embedding_size+n_nodes*atom_dist]
            
        output:
            AVG_MAT (tensor) : [batch_size,n_nodes, n_nodes] tensor
        """
        B,N_residu, _ = Fh.shape
        # Calculate the pairwise avrege between each node in the tensor
        pairwise_avg = (Fh.unsqueeze(axis=2) + Fh.unsqueeze(axis=1))/2   
        return torch.sum(pairwise_avg,axis=-1)
    
    def get_Grad_mat(self,Fh):
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
        pairwise_differences = (Fh.unsqueeze(axis=2) - Fh.unsqueeze(axis=1))/self.h
        # Calculate the pairwise squared distances between each node in the tensor
        # pairwise_squared_distances = torch.sum(pairwise_differences**2, axis=-1)
        # # Calculate the pairwise distances between each node in the tensor
        # distances = torch.sqrt(pairwise_squared_distances)
        return torch.sum(pairwise_differences,axis=-1)
    
class PEM(torch.nn.Module):
  """Protein energy model"""
  
  def __init__(self, dim_in, dim_h, dim_out, layers, model_type, gs_coef,heads = 8):
    super().__init__()
    
    if model_type == 'GCN':
      self.model = [GCN(dim_in, dim_h, dim_out) for i in range(layers)]
    elif model_type == 'GAT':
      # self.model = [GAT(dim_in, dim_h, dim_out) for i in range(layers)]
      self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
      self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
      self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.005,
                                        weight_decay=5e-4)
      self.gs_coef = gs_coef
    else:
      raise ValueError('Model type not supported')
    self.layers = layers
    # First fully connected layer
    self.fcs1 = nn.Linear(36, 64)
    self.fcs2 = nn.Linear(64, 36)
    self.bn1  = nn.BatchNorm1d(36)
    self.bn2  = nn.BatchNorm1d(36)
    # First fully connected layer
    self.fc1 = nn.Linear(36, 64)
    # Second fully connected layer that outputs our 10 labels
    self.fc2 = nn.Linear(64, 1)
  
  def forward(self,x_decoy, emb_decoy,x_native,emb_native ,edge_index):
      x_decoy  = self.get_graph(x_decoy, emb_decoy)
      identity = x_decoy
      x = x_decoy
      x = self.fcs1(x)
      x = F.relu(x)
      x = self.fcs2(x)
      x = self.bn1(x)
      for layer in range(self.layers):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        
        h = F.log_softmax(h, dim=1)+identity
      x = self.bn2(x)
      x  = self.fc1(x)
      x = F.relu(x)
      x_decoy = self.fc2(x)

      
      x_native  = self.get_graph(x_native, emb_native)
      identity = x_native
      x = x_native
      x = self.fcs1(x)
      x = F.relu(x)
      x = self.fcs2(x)
      x = self.bn1(x)
      for layer in range(self.layers):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        
        h = F.log_softmax(h, dim=1)+identity
      x = self.bn2(x)
      x  = self.fc1(x)
      x = F.relu(x)
      x_native = self.fc2(x)
     
      return torch.cat((self.get_energy(x_decoy).unsqueeze(0), self.get_energy(x_native).unsqueeze(0)),dim=0)
    
  def get_graph(self,x, emb):
    """Get graph representation of protein"""
    D = self.get_dist_matrix(x) # N,N,16
    D = torch.relu(torch.exp(self.gs_coef*D**2))
    
    D = D.sum(dim=1) #N,16
    
    Fh = torch.cat([emb,D],dim=1) #N,16+emb_size
    
    return Fh
  
  def get_dist_matrix(self,Xd):
      """
      Return the node distence matrix
      Args:
          Xd (tensor):X embeded [n_nodes ,num_atoms=4,new_cords_size]
      Returns:
          tensor : [n_nodes,n_nodes ,atom_dist=16] tensor
      """
      N_residu,N_atoms,coords_size = Xd.shape
      Xd = Xd.reshape(N_residu*N_atoms,coords_size)
      D = torch.cdist(Xd,Xd,p=2)
      D = D.reshape(N_residu,N_atoms,N_residu,N_atoms)
      D = torch.swapaxes(D,1,2)
      D = D.reshape(N_residu,N_residu,N_atoms*N_atoms)
      return D
  
  def get_energy(self,Fh):
        """
        Calculates the energy of the protein
        Inputs:
            Fh: a [n_nodes , embedding_size+N_residu] tensor
        Returns:
            Energy [batch_size] tensor
        """
        E = torch.sum(Fh**2,dim=(0,1))
        return E