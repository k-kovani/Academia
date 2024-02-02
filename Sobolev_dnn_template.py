import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import KFold
from tensorflow.keras.models import clone_model



# ============================
# Define the Model:
# ============================
class MyModel(tf.keras.Model):
 def __init__(self):
  super().__init__()
  
  # Function Layers:
  self.dense1=tf.keras.layers.Dense(32,activation='gelu')
  self.dense2=tf.keras.layers.Dense(64,activation='gelu')
  self.dense3=tf.keras.layers.Dense(64,activation='gelu')
  self.dense4=tf.keras.layers.Dense(32,activation='gelu')
  self.dense5=tf.keras.layers.Dense(32,activation='gelu')
  self.dense6=tf.keras.layers.Dense(1,activation='tanh')
  
 
 def call(self, inputs):
    
     with tf.GradientTape(persistent=True) as t:
            
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            t.watch(inputs_tensor)
            
            # Forward Pass:
            h=self.dense1(inputs_tensor)
            h=self.dense2(h)
            h=self.dense3(h)
            h=self.dense4(h)
            h=self.dense5(h)
            h=self.dense6(h)
     
     # Compute gradient:
     dh = t.gradient(h, inputs_tensor)

     return h, dh


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
 
