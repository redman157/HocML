from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Activation
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
import pickle
import os
%matplotlib inline
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
reader = pd.read_csv("training.csv")
records = reader.Image
inputs = reader.iloc[:,:-2]

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

