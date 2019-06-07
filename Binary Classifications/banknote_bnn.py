# banknote_bnn.py
# Banknote classification
# Keras 2.1.5 TensorFlow 1.7.0 Anaconda3 4.1.1
# raw data looks like:
# 4.5459,8.1674,-2.4586,-1.4621,0
# -1.3971,3.3191,-1.3927,-1.9948,1
# 0 = authentic, 1 = fake

import numpy as np
import keras as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU msg

class MyLogger(K.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss =logs.get('loss')
      curr_acc = logs.get('acc') * 100
      print("epoch = %4d  loss = %0.6f  acc = %0.2f%%" % \
        (epoch, curr_loss, curr_acc))

def main():
  print("\nBanknote authentication dataset example \n")
  np.random.seed(1)

  # 1. load data
  # print("Loading data into memory ")
  train_file = "/home/pawan/Downloads/Binary Classifications/banknote_train_mm_tab.txt"
  test_file = "/home/pawan/Downloads/Binary Classifications/banknote_test_mm_tab.txt"

  train_x = np.loadtxt(train_file, delimiter='\t',
    usecols=[0,1,2,3], dtype=np.float32)
  train_y = np.loadtxt(train_file, delimiter='\t',
    usecols=[4], dtype=np.float32)
  test_x = np.loadtxt(test_file, delimiter='\t', 
    usecols=[0,1,2,3], dtype=np.float32)
  test_y =np.loadtxt(test_file, delimiter='\t',
    usecols=[4], dtype=np.float32)

  # 2. define 4-(x-x)-1 deep NN model
  print("Creating 4-(8-8)-1 binary NN classifier \n")
  my_init = K.initializers.glorot_uniform(seed=1)
  model = K.models.Sequential()
  model.add(K.layers.Dense(units=8, input_dim=4,
    activation='tanh', kernel_initializer=my_init)) 
  model.add(K.layers.Dense(units=8, activation='tanh',
    kernel_initializer=my_init)) 
  model.add(K.layers.Dense(units=1, activation='sigmoid',
    kernel_initializer=my_init)) 

  # 3. compile model
  simple_sgd = K.optimizers.SGD(lr=0.01)  
  model.compile(loss='binary_crossentropy',
    optimizer=simple_sgd, metrics=['accuracy'])  

  # 4. train model
  max_epochs = 500
  my_logger = MyLogger(n=50)
  h = model.fit(train_x, train_y, batch_size=32,
    epochs=max_epochs, verbose=0, callbacks=[my_logger]) 

  # 5. evaluate model
  np.set_printoptions(precision=4, suppress=True)
  eval_results = model.evaluate(test_x, test_y, verbose=0) 
  print("\nLoss, accuracy on test data: ")
  print("%0.4f %0.2f%%" % (eval_results[0], \
eval_results[1]*100))

  # 6. save model
  # mp = ".\\Models\\banknote_model.h5"
  # model.save(mp)

  # 7. make a prediction
  inpts = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
  pred = model.predict(inpts)
  print("\nPredicting authenticity for: ")
  print(inpts)
  print("Probability that class = 1 (fake):")
  print(pred)

if __name__=="__main__":
  main()
