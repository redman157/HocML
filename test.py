from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D,Dropout,Dense, Flatten, Activation
from keras.models import model_from_json,Model
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
  if os.path.exists("train") is False:
    data = load_dataset()
    with open("train","wb") as f:
      pickle.dump(load_dataset(),f)
    f.close()
  else:
    with open("train","rb") as f:
      data = pickle.load(f)
    f.close()
  return data
X_train = load_data()
X_train = X_train.reshape(-1,96,96)
#y_train = inputs
training = reader_train.drop('Image',axis = 1)
y_train = []
for i in range(records_test.shape[0]):
  y = training.iloc[i,:]
  y_train.append(y)
y_train = np.array(y_train,dtype = 'float')
print('X_train' , X_train.shape)
print('y_train' , y_train.shape)
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
model.fit(X_train,y_train,epochs = 50)

def load_testset():
  X_test = None
  for i in range(record_test.shape[0]):
    if X_test is None :
      X_test = np.array(record[i].split(' '),dtype = np.int32)
    else:
      X_test = np.concatenate((X_test,np.array(record[i].split(' '),dtype = np.int32)),axis = 0)
  else:
    return X_test
X_test = load_testset()
X_test = X_test.reshape(-1,96,96)

# luu mode vua train vao json
model_json = model_from_json() 
with open('model_json','w') as file:
  file.write(model_json)
model.save_weights("model.h5")
print("save model vao o dia")

file = open('model_json','r')
load_model_json = file.read()
file.close()
loaded_model = model_from_json(load_model_json)
loaded_model.load_weights('model.h5')
print("load model trong o dia")
loaded_model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mae','accuracy'])
