# ======================
# import libraries:
# ======================

import os
import time
import numpy as np
import scipy.sparse as sp
import trimesh

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import mean_absolute_error

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GraphSageConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.normalize_one import NormalizeOne



# ========================
# Create custom dataset:
# ========================

class CustomDataset(Dataset):

    def __init__(self, files_path, n_samples, min_nodes=None, max_nodes=None, n_features=3, **kwargs):
        self.f_path = files_path        # path of the dataset folder (mesh entities in .obj files).
        self.n_samples = n_samples      # number of samples to collect from the dataset.
        self.n_features = n_features    # number of graph node features. Default is 3: (x, y, z) coordinates of mesh vertices.
        self.min_nodes = min_nodes      # min. number of graph nodes (mesh vertices).
        self.max_nodes = max_nodes      # max. number of graph nodes (mesh vertices).
        super().__init__(**kwargs)

    def create_adjacency_from_faces(self, num_vertices, faces):

      # initialize the adjacency matrix:
      adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

      # Populate the adjacency matrix based on faces:
      for face in faces:
          for i in range(3):
              # Connect each vertex to the other two in the face:
              v1, v2 = face[i], face[(i + 1) % 3]
              adjacency_matrix[v1, v2] = 1
              adjacency_matrix[v2, v1] = 1

      # Transform the adjacency matrix to sparse:
      a = sp.csr_matrix(adjacency_matrix)
      return a


    def read(self):
        def make_graph(file):
            # Read the .obj file from path and create a trimesh mesh entity:
            mesh = trimesh.load(f'{self.f_path}/{file}')

            # Check if the number of vertices (nodes) falls within the specified range:
            num_vertices = len(mesh.vertices)
            if (self.min_nodes is not None and num_vertices < self.min_nodes) or \
               (self.max_nodes is not None and num_vertices > self.max_nodes):
                return None  

            # Node features
            x = mesh.vertices  # shape: (n_nodes, n_features)

            # Adjacency:
            faces = mesh.faces
            a = self.create_adjacency_from_faces(num_vertices, faces) # shape: (n_nodes, n_nodes)

            # Labels:
            # vol = mesh.volume
            vol_ratio = mesh.volume / mesh.convex_hull.volume

            y = np.array([vol_ratio]).reshape(1,)  # shape: (num_target, )

            return Graph(x=x, a=a, y=y)

        # Return a list of Graph objects, excluding None values:
        return [graph for graph in (make_graph(file) for file in os.listdir(self.f_path)[:self.n_samples]) if graph is not None]


# ===============================================
# Create the dataset and set the loaders:
# ===============================================
file_path = 'path/to/DB/folder'
n_samples = int(1000)
min_nodes = 500
max_nodes = 1000

# Dataset:
# NormalizeAdj(): normalizes the adjacency matrix as A ← D^(−1/2) A D^(−1/2).
# NormalizeOne(): normalizes the node attributes by dividing each row by its sum, so that it sums to 1.
dataset = CustomDataset(file_path, n_samples, min_nodes, max_nodes, transforms=[NormalizeAdj(), NormalizeOne()])

# Train/valid/test split:
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = abc_dataset[idx_tr]
data_va = abc_dataset[idx_va]
data_te = abc_dataset[idx_te]

# Data loaders:
batch_size= len(dataset)
epochs=2000
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(data_te, batch_size=batch_size)



# ===============================================
# Build the GCN model, train and evaluate:
# ===============================================

class MyGCN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphSageConv(64, activation="relu")
        self.conv2 = GraphSageConv(256, activation="relu")
        self.conv3 = GraphSageConv(32, activation="relu")
        self.conv4 = GraphSageConv(64, activation="relu")
        self.conv5 = GraphSageConv(64, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(abc_dataset.n_labels, activation="tanh")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.conv4([x, a])
        x = self.conv5([x, a])
        pool = self.global_pool([x, i])
        output = self.dense(pool)

        return output


model = MyGCN()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer,loss=loss_fn)

# Train the model:
start_time = time.time()
history = model.fit(
          loader_tr.load(),
          steps_per_epoch=loader_tr.steps_per_epoch,
          epochs=epochs,
          validation_data=loader_va.load(),
          validation_steps=loader_va.steps_per_epoch
          )

end_time = time.time()
print('\nTotal Computation time: {} seconds\n'.format(round(end_time - start_time)))


# Evaluate the model:
# preds = model.evaluate(
#         loader_te.load(),
#         steps=loader_te.steps_per_epoch
#         )

all_predictions = []
all_targets = []

for batch in loader_te.load():
    inputs, targets = batch
    predictions = model(inputs, training=False)
    all_predictions.append(predictions)
    all_labels.append(targets)
