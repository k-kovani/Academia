# =======================
# Import the libraries:
# =======================

import os
import time
import trimesh

# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

# model, train function and evaluate function:
from pylightning_model import gnn_layer_by_name, GNNModel, GraphGNNModel, GraphLevelGNN
from pylightning_model import train_model, evaluate

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 50
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

# Setting the seed
pl.seed_everything(42)


# ===================================
# Create Custom dataset:
# ===================================

class CustomDataset(Dataset):
  
    def __init__(self, files_path, n_samples, min_nodes=None, max_nodes=None, n_features=3, transform=None):
        self.f_path = files_path           # path of the dataset folder (mesh entities in .obj files).
        self.n_samples = n_samples         # number of samples to collect from the dataset.
        self.n_features = n_features       # number of graph node features. Default is 3: (x, y, z) coordinates of mesh vertices.
        self.transform = transform         # transformations.
        self.min_nodes = min_nodes         # min. number of graph nodes (mesh vertices).
        self.max_nodes = max_nodes         # max. number of graph nodes (mesh vertices).
        super().__init__()

    def create_adjacency_from_faces(self, num_vertices, faces):
        
        # initialize the adjacency matrix:
        adjacency_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float)
      
        # Populate the adjacency matrix based on faces:
        for face in faces:
            for i in range(3):
                # Connect each vertex to the other two in the face:
                v1, v2 = face[i], face[(i + 1) % 3]
                adjacency_matrix[v1, v2] = 1
                adjacency_matrix[v2, v1] = 1

        return adjacency_matrix

    def read(self, file):
      
        # Load trimesh mesh
        mesh = trimesh.load(f'{self.f_path}/{file}')

        # Check if the number of nodes falls within the specified range:
        num_vertices = len(mesh.vertices)
        if (self.min_nodes is not None and num_vertices < self.min_nodes) or \
           (self.max_nodes is not None and num_vertices > self.max_nodes):
            return None  # Skip this graph if it doesn't meet the node count criteria

        # Node features
        x = torch.tensor(mesh.vertices, dtype=torch.float)

        # Adjacency 
        num_vertices = len(x)
        faces = mesh.faces
        a = self.create_adjacency_from_faces(num_vertices, faces)

        # Labels
        # vol = mesh.volume
        vol_ratio = mesh.volume / mesh.convex_hull.volume

        y = torch.tensor([vol_ratio], dtype=torch.float).view(1,)

        return Data(x=x, edge_index=a.nonzero(as_tuple=False).t().contiguous(), y=y)

    def __len__(self):
        files = os.listdir(self.f_path)
        return min(len(files), self.n_samples)

    def __getitem__(self, index):
        files = os.listdir(self.f_path)[:self.n_samples]

        if isinstance(index, slice):
            files = files[index]
            return [self.read(file) for file in files if self.read(file) is not None]

        file = files[index]
        data = self.read(file)

        if self.transform:
            data = self.transform(data)

        return data


# =================================
# Create the dataset:
# =================================
file_path = 'path/to/DB/folder'
n_samples = int(1000)
min_nodes = 500
max_nodes = 1000

# Dataset:
dataset = CustomDataset(file_path, n_samples, min_nodes, max_nodes)

# Train/valid/test split:
train_dataset = dataset
val_dataset = dataset[0:2]
test_dataset = dataset[2:4]

# Loaders:
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# =======================================
# Build the model, train and evaluate:
# =======================================

model = GraphLevelGNN(c_in=3, c_hidden=64, c_out=1)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train:
num_epochs = 1000
history = train_model(model, train_loader, criterion, optimizer,\
                      num_epochs=num_epochs, verbose=True, \
                      val_loader=val_loader)

# Evaluate:
predictions = evaluate(model, test_loader, save_results=True)

