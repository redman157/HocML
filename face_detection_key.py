from tensorflow import keras
from keras.layers import Dense
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
%matplotlib inline
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
reader = pd.read_csv("training.csv")
records = reader.Image
inputs = reader.iloc[:,:-2]

records = records.values
print(records.shape)
inputs = inputs.values
images = None
for i in range(records.shape[0]):
    if images is None:
        images = np.array(records[i].split(' '), dtype=np.int32)
    else:
        images = np.concatenate((images, np.array(records[i].split(' '), dtype=np.int32)), axis=0)
else:
    print(images.reshape(96,96,1))
    print(inputs.shape)
