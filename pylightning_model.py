# =======================
# Import the libraries:
# =======================

import os
import time

# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data

# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor


AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 50
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

# Setting the seed
pl.seed_everything(42)


# ======================================================
# Model: GNNModel --> GraphGNNModel --> GraphLevelGNN
# ======================================================

# import layers by name:
gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class GraphGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of output features (usually number of classes)
            dp_rate_linear: Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs: Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in, c_hidden=c_hidden, c_out=c_hidden, **kwargs)  # Not our prediction output yet!
        self.head = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out))

    def forward(self, x, edge_index, batch_idx):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx: Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)  # Average pooling
        x = self.head(x)
        x = torch.sigmoid(x)
        return x



class GraphLevelGNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        return x

    def training_step(self, data):
        x_out = self.forward(data)
        loss = nn.MSELoss(x_out, data.y)

        # metrics here:
        self.log("loss/train", loss)
        return loss


    def validation_step(self, data, batch_idx):
        x_out = self.forward(data)
        loss = nn.MSELoss(x_out, data.y)

        # metrics here:
        self.log("loss/validation", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 3e-4, weight_decay=0.0)


# =========================
# Evaluate:
# =========================

def evaluate(model, test_loader, save_results=True, tag="_default", verbose=False):

    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.MSELoss()

    model.eval()
    total_loss = 0
    total_batches = 0

    for batch in test_loader:

        pred = model(batch.to(my_device))
        loss = criterion(pred, batch.y.to(my_device))
        total_loss += loss.detach()
        total_batches += batch.batch.max()

    test_loss = total_loss / total_batches

    results = {
        "loss": test_loss, \
        "tag": tag }

    return results



# =========================
# Train model:
# =========================
T_loss = []
V_loss = []

def train_model(model, train_loader, criterion, optimizer, num_epochs=1000, \
        verbose=True, val_loader=None, save_tag="default_run_"):


    ## call validation function and print progress at each epoch end
    display_every = 1 #num_epochs // 10
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(my_device)

    # we'll log progress to tensorboard
    log_dir = f"lightning_logs/plain_model_{str(int(time.time()))[-8:]}/"
    writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    for epoch in range(num_epochs):

        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad()

            pred = model(batch.to(my_device))
            loss = criterion(pred, batch.y.to(my_device))
            loss.backward()

            optimizer.step()

            total_loss += loss.detach()
            batch_count += 1

        mean_loss = total_loss / batch_count


        writer.add_scalar("loss/train", mean_loss, epoch)

        if epoch % display_every == 0:
            train_results = evaluate(model, train_loader, \
            tag=f"train_ckpt_{epoch}_", verbose=True)
            train_loss = train_results["loss"]



        if val_loader is not None:
            val_results = evaluate(model, val_loader, \
            tag=f"val_ckpt_{epoch}_", verbose=True)
            val_loss = val_results["loss"]


        T_loss.append(train_loss)
        V_loss.append(val_loss)


        print(f"Epoch: {epoch},  Train loss: {train_loss},  Val_loss: {val_loss}")

