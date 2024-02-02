import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import KFold
from tensorflow.keras.models import clone_model


# ===================================
# Create custom weight initializer:
# ===================================
class Weights_init(tf.keras.initializers.Initializer):
 def __init__(self, target_values):
  self.b = target_values
              
 def __call__(self, shape, dtype=None, **kwargs):
    return tf.convert_to_tensor(self.b, dtype)

 def get_config(self): 
    return {'b': self.b}


# ==================
# Define the model:
# ==================
class MyModel(tf.keras.Model):
 def __init__(self, Fs, dFs):
  super().__init__()

  self.Ys = Fs
  self.dYs = dFs
  self.Nb = dFs.shape[1]
  self.Ns = dFs.shape[0]
  
  # Function Branch Layers:
  self.dense1=tf.keras.layers.Dense(32,activation='gelu')
  self.dense2=tf.keras.layers.Dense(128,activation='gelu')
  self.dense3=tf.keras.layers.Dense(64,activation='gelu')
  self.dense4=tf.keras.layers.Dense(self.Ns,activation='gelu')
  # Function-Basis Layer:
  weights_init = Weights_init(self.Ys.reshape((self.Ns, 1))) 
  self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weights_init, trainable=False)
  
  
  # Gradient Branch Layers:
  self.gdense1=tf.keras.layers.Dense(64,activation='gelu')
  self.gdense2=tf.keras.layers.Dense(32,activation='gelu')
  self.gdense3=tf.keras.layers.Dense(256,activation='gelu')
  self.gdense4=tf.keras.layers.Dense(64,activation='gelu')
  self.gdense5=tf.keras.layers.Dense(32,activation='gelu')
  self.gdense6=tf.keras.layers.Dense(128,activation='gelu')
  # Gradient-Basis Layer:
  weights_init = Weights_init(self.dYs.reshape((self.Ns, self.Nb))) 
  self.gdense7 = tf.keras.layers.Dense(self.Nb, activation='sigmoid', kernel_initializer=weights_init,trainable=False)


 
 def call(self, inputs):
    
     with tf.GradientTape(persistent=True) as t:
            
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            t.watch(inputs_tensor)
            
            # Forward pass - function layers:
            h=self.dense1(inputs_tensor)
            h=self.dense2(h)
            h=self.dense3(h)
            h=self.dense4(h)
            h=self.dense5(h)
            B_func = h


            # Forward pasdd - gradient layers:
            h_hat=self.gdense1(inputs_tensor)
            h_hat=self.gdense2(h_hat)
            h_hat=self.gdense3(h_hat)
            h_hat=self.gdense4(h_hat)
            h_hat=self.gdense5(h_hat)
            h_hat=self.gdense6(h_hat)
            h_hat=self.gdense7(h_hat)
            B_grad = h_hat

            # Compute Hermite interpolation function:      
            g = B_func + tf.expand_dims(tf.math.reduce_sum(B_grad, axis=1) , axis=-1)
     
     # Compute the gradient:
     dg = t.gradient(g, inputs_tensor)

     return g, dg



# ==============================
# Define the training function:
# ==============================
def train(data):
 
   x, y, dy = data
    
   "x: Design Variables"
   "y: Objective Function"
   "dy: Sensitivity Derivatives"
   
   nSamples = x.shape[0]       # Get the number of samples
   Nb = x.shape[1]             # Get the number of Design Variables
   nVal = int(0.2 * nSamples)  # Split 20% of samples for validation
   nTPs = nSamples-nVal        # Create the training patterns
   
   # set the device:
   gpu_id=2
   os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    
   # Instatiate & compile the model:
   model=MyModel()
   optim = tf.keras.optimizers.Adam(learning_rate=0.001)
   loss_fn = tf.keras.losses.MeanSquaredError()
   metrics = tf.keras.metrics.MeanAbsoluteError()
   model.compile(optimizer=optim,
                  loss = [loss_fn, loss_fn],
                  loss_weights = [1.0, 1.0],
                  metrics = [metrics]
   )
   
   # Create callback:
   ckpt=tf.train.get_checkpoint_state('checkpoint/')
   if ckpt is not None:
        try:
            model.load_weights('checkpoint/save')
        except:
            print('error in loading checkpoint, random init will be used instead')
   
   callback = tf.keras.callbacks.ModelCheckpoint('checkpoint/save', save_best_only=False, save_weights_only=True,
                                                  verbose=0, save_freq=100)
   
   
   ### Train with Cross-Validation ###
   nCycles = 5  # Number of cycles
   nFolds = 5   # Number of folds for cross-validation
   nBatch = nSamples
   nEpoch = 1000
   
   # Create KFold instance
   kf = KFold(n_splits=nFolds, shuffle=True, random_state=42)
   
   total_history = []  # Storing List 
   
   # Iterate over cycles
   for iCyc in range(nCycles):
       for train_index, val_index in kf.split(range(nSamples)):
           
           # Create a new instance of the model for each fold
           model_copy = clone_model(model)
           model_copy.compile(optimizer=optim, loss=[loss_fn, loss_fn], metrics=[metrics])
   
           # Split the data into training and validation sets
           x_train, y_train, dy_train = x[train_index], y[train_index], dy[train_index]
           x_val, y_val, dy_val = x[val_index], y[val_index], dy[val_index]
   
           # Train the model
           history = model_copy.fit(x=x_train, y=[y_train, dy_train], batch_size=nBatch, epochs=nEpoch, verbose=2, callbacks=callback,
                                    validation_data=(x_val, [y_val, dy_val]), validation_freq=1)
   
           # Store the history for each fold
           total_history.append(history)
    
   # save the model:
   model.save("./saved_model")
   
   return total_history
 

