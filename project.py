from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D,Dropout,Dense, Flatten, Activation
from keras.models import model_from_json,Model, load_model, save_model
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
import pickle
import h5py
import os
%matplotlib inline
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
reader_train = pd.read_csv("training.csv")
reader_test = pd.read_csv("test.csv")
reader_Id = pd.read_csv("IdLookupTable.csv")
reader_train.isnull().any().value_counts()
reader_train.fillna(method = 'ffill',inplace=True)

records = reader_train.Image
inputs = reader_train.iloc[:,:-2]

records_test = reader_test.Image

records = records.values
inputs = inputs.values

def load_dataset():
  images = None
  for i in range(0,records.shape[0]):
    if images is None:
      images = np.array(records[i].split(' '), dtype=np.int32)
    else:
      images = np.concatenate((images, np.array(records[i].split(' '), dtype=np.int32)), axis=0)
  else:
    return images
def load_data():
  if os.path.exists("train_x") is False:
    data = load_dataset()
    with open("train_x","wb") as f:
      pickle.dump(load_dataset(),f)
    f.close()
  else:
    with open("train_x","rb") as f:
      data = pickle.load(f)
    f.close()
  return data
X_train = load_data()
X_train = X_train.reshape(-1,96,96)
#y_train = inputs

training = reader_train.drop('Image',axis = 1)
y_train = []
for i in range(records.shape[0]):
  y = training.iloc[i,:]
  y_train.append(y)
y_train = np.array(y_train,dtype = 'float')
print('X_train' , X_train.shape)
print('y_train' , y_train.shape)

def load_testset():
  X_test = None
  for i in range(records_test.shape[0]):
    if X_test is None :
      X_test = np.array(records_test[i].split(' '),dtype = np.int32)
    else:
      X_test = np.concatenate((X_test,np.array(records_test[i].split(' '),dtype = np.int32)),axis = 0)
  else:
    return X_test

def load_datatest():
  if os.path.exists("test_x") is False:
    data = load_dataset()
    with open("test_x","wb") as f:
      pickle.dump(load_testset(),f)
    f.close()
  else:
    with open("test_x","rb") as f:
      data = pickle.load(f)
    f.close()
  return data
X_test = load_datatest()
X_test = X_test.reshape(-1,96,96)
print("X_test", X_test.shape)
model =  Sequential()
# x = (7049,96,96)
# y = (7049,30)
model.add(Flatten(input_shape= (96,96)))
model.add(Dense(128,activation = "relu"))
model.add(Dropout(0.1))

model.add(Dense(64,activation = "relu"))
model.add(Dense(30))

model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mae','accuracy'])
model.fit(X_train,y_train,epochs = 50, batch_size = 128,validation_split = 0.2)
model.save('model.h5')

json_string = model.to_json()
model = model_from_json(json_string)
model.load_weights('model.h5',by_name = True)
model.load_model('model.h5')
def loaded_model():
  model = load_model('model.h5')
  return model
def show_results(images_index):
  pred = model.predict(X_test[images_index:(images_index+1)])
  show_images(X_test[images_index], pred[0])
show_images(3)

