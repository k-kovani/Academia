import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from keras import backend as K


class Weights_init(tf.keras.initializers.Initializer):
 def __init__(self, target_values):
  self.b = target_values
              
 def __call__(self, shape, dtype=None, **kwargs):
    return tf.convert_to_tensor(self.b, dtype)

 def get_config(self): 
    return {'b': self.b}


class MyModel(tf.keras.Model):
 def __init__(self):
  super().__init__()
  
  # Function Layers:
  self.dense1=tf.keras.layers.Dense(2048,activation='gelu')
  self.dense2=tf.keras.layers.Dense(4096,activation='gelu')
  self.dense3=tf.keras.layers.Dense(1024,activation='gelu')
  self.dense4=tf.keras.layers.Dense(256,activation='gelu')
  self.dense5=tf.keras.layers.Dense(1024,activation='gelu')
  self.dense6=tf.keras.layers.Dense(1,activation='tanh')
  
 
 def call(self, inputs):
    
     with tf.GradientTape(persistent=True) as t:
            
            inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            t.watch(inputs_tensor)
            
            # Feedforward function layers:
            h=self.dense1(inputs_tensor)
            h=self.dense2(h)
            h=self.dense3(h)
            h=self.dense4(h)
            h=self.dense5(h)
            h=self.dense6(h)
     
     # Hermite interpolation gradient:
     dh = t.gradient(h, inputs_tensor)

     return h, dh



def train(data):
 
 x, (y, dy) = data

 nSamples = x.shape[0]
 Nb = x.shape[1]
 nVal = int(0.2 * nSamples) # Split 20% for Validation
 nTPs = nSamples-nVal

 x_db = np.zeros((nTPs,Nb))
 y_db = np.zeros((nTPs,1 ))
 dy_db = np.zeros((nTPs, Nb))
 x_val= np.zeros((nVal,Nb))
 y_val= np.zeros((nVal,1 ))
 dy_val = np.zeros((nVal, Nb))
      
 indList = np.arange(nSamples)
 
 gpu_id=2
 os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
 model=MyModel()
 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss = tf.keras.losses.MeanSquaredError()
 )
 ckpt=tf.train.get_checkpoint_state('checkpoint/')
 if ckpt is not None:
     try:
         model.load_weights('checkpoint/save')
     except:
         print('error in loading checkpoint, random init will be used instead')

 callback = tf.keras.callbacks.ModelCheckpoint('checkpoint/save', save_best_only=False, save_weights_only=True,
                                               verbose=0, save_freq=100)



 nCycles = 1
 nBatch = x_db.shape[0]
 nEpoch = 200

 total_history = []
 for iCyc in range(nCycles):
    np.random.shuffle(indList)
    ks=0
    for ind in range(nTPs):
        isam = indList[ind]
        x_db[ks,:] = x[isam,:]
        y_db[ks,:] = y[isam,:]
        dy_db[ks,:] = dy[isam,:]
        ks +=1
    
    kv=0
    for ind in range(nTPs,nSamples):
        isam = indList[ind]
        x_val[kv,:] = x[isam,:]
        y_val[kv,:] = y[isam,:]
        dy_val[kv,:] = dy[isam, :]
        kv +=1
 
    history=model.fit(x=x_db, y=[y_db, dy_db], batch_size=nBatch, epochs=nEpoch, verbose=2, callbacks=callback,
            validation_data=(x_val,[y_val, dy_val]), validation_freq=1)

    total_history.append(history)
 

 model.save("./saved_model")


 return total_history
 




# ----------------
# Train the model:
# ----------------
DB = np.load("train_DB_non_dim.npy")
N = DB.shape[0]
Input = DB[:, 0:2].reshape((N, 2))
Output = DB[:, 2].reshape((N, 1))
gradDB = DB[:, 3:5].reshape((N, 2))



data = (Input, (Output, gradDB))
predicted  = train(data)

