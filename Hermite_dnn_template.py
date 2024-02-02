import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


class Weights_init(tf.keras.initializers.Initializer):
 def __init__(self, target_values):
  self.b = target_values
              
 def __call__(self, shape, dtype=None, **kwargs):
    return tf.convert_to_tensor(self.b, dtype)

 def get_config(self): 
    return {'b': self.b}


class MyModel(tf.keras.Model):
 def __init__(self, nSamples, nDV, Ysamples, dYsamples):
  super().__init__()
  self.Nb = nDV
  self.Ns = nSamples
  self.Ys = Ysamples
  self.dYs = dYsamples
  
  # Function Branch Layers:
  self.dense1=tf.keras.layers.Dense(2048,activation='gelu')
  self.dense2=tf.keras.layers.Dense(4096,activation='gelu')
  self.dense3=tf.keras.layers.Dense(128,activation='gelu')
  self.dense4=tf.keras.layers.Dense(self.Ns,activation='gelu')
  # Function-Basis Layer:
  weights_init = Weights_init(self.Ys.reshape((self.Ns, 1))) 
  self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weights_init, trainable=False)
  
  
  # Gradient Branch Layers:
  self.gdense1=tf.keras.layers.Dense(64,activation='gelu')
  self.gdense2=tf.keras.layers.Dense(32,activation='gelu')
  self.gdense3=tf.keras.layers.Dense(4096,activation='gelu')
  self.gdense4=tf.keras.layers.Dense(64,activation='gelu')
  self.gdense5=tf.keras.layers.Dense(32,activation='gelu')
  self.gdense6=tf.keras.layers.Dense(2048,activation='gelu')
  self.gdense7=tf.keras.layers.Dense(128,activation='gelu')
  self.gdense8=tf.keras.layers.Dense(64,activation='gelu')
  self.gdense9=tf.keras.layers.Dense(1024,activation='gelu')
  self.gdense10=tf.keras.layers.Dense(self.Ns,activation='gelu')
  # Gradient-Basis Layer:
  weights_init = Weights_init(self.dYs.reshape((self.Ns, self.Nb))) 
  self.gdense11 = tf.keras.layers.Dense(self.Nb, activation='tanh', kernel_initializer=weights_init,trainable=False)


 
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
            B_func = h


            # Feedforward gradient layers:
            h_hat=self.gdense1(inputs)
            h_hat=self.gdense2(h_hat)
            h_hat=self.gdense3(h_hat)
            h_hat=self.gdense4(h_hat)
            h_hat=self.gdense5(h_hat)
            h_hat=self.gdense6(h_hat)
            h_hat=self.gdense7(h_hat)
            h_hat=self.gdense8(h_hat)
            h_hat=self.gdense9(h_hat)
            h_hat=self.gdense10(h_hat)
            h_hat=self.gdense11(h_hat)
            B_grad = h_hat

            # Hermite interpolation:      
            g = B_func + tf.expand_dims(tf.math.reduce_sum(B_grad, axis=1) , axis=-1)
     
     # Hermite interpolation gradient:
     dg = t.gradient(g, inputs_tensor)

     return g, dg



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

 # Define the model:
 model=MyModel(nSamples=nSamples, nDV=Nb, Ysamples=y, dYsamples=dy)

 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss = tf.keras.losses.MeanSquaredError(), 
               #loss = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()]
               #loss_weights = [1.0, 1.0]
               #run_eagerly=True
 )
 ckpt=tf.train.get_checkpoint_state('checkpoint/')
 if ckpt is not None:
     try:
         model.load_weights('checkpoint/save')
     except:
         print('error in loading checkpoint, random init will be used instead')

 callback = tf.keras.callbacks.ModelCheckpoint('checkpoint/save', save_best_only=False, save_weights_only=True,
                                               verbose=0, save_freq=100)



 nCycles = 5
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
Input = np.load("Input.npy")
Output = np.load("Output.npy")
gradDB = np.load("gradDB.npy")

data = (Input, (Output, gradDB))
predicted = train(data)


# Save Metrics:
# =============
nCycles = len(predicted)
nEpoch = 200

loss = np.zeros((nCycles*nEpoch, ))
val_loss = np.zeros((nCycles*nEpoch, ))
floss = np.zeros((nCycles*nEpoch, ))
gloss = np.zeros((nCycles*nEpoch, ))
fval_loss = np.zeros((nCycles*nEpoch, ))
gval_loss = np.zeros((nCycles*nEpoch, ))
istart, iend = 0, nEpoch
for i in range(nCycles):
      loss[istart:iend] = predicted[i].history["loss"]
      val_loss[istart:iend] = predicted[i].history["val_loss"]
      floss[istart:iend] = predicted[i].history["func_loss"]
      fval_loss[istart:iend] = predicted[i].history["val_func_loss"]
      gloss[istart:iend] = predicted[i].history["grad_loss"]
      gval_loss[istart:iend] = predicted[i].history["val_grad_loss"]
      istart += nEpoch
      iend += nEpoch

np.savetxt("./Metrics/loss", loss)
np.savetxt("./Metrics/val_loss", val_loss)
np.savetxt("./Metrics/func_loss", floss)
np.savetxt("./Metrics/val_func_loss",fval_loss)
np.savetxt("./Metrics/grad_loss", gloss)
np.savetxt("./Metrics/val_grad_loss", gval_loss)







