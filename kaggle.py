from __future__ import division, print_function
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os 
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder
validation_size = 0.20 # phuong size cua validation = 20%
seed = 7
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

filename = (path.split('/')[-1])
dir = os.path.join(os.path.abspath(os.path.dirname(filename)))
r = requests.get(path,allow_redirects=True, stream = True,verify = False)
if r.status_code == 200:    
  with open(dir + '/' + filename ,'wb') as f:
    f.write(r.content)
cols = ['SepalLength','SepalWidth','PetalLength', 'PetalWidth',"Class"] 
data = pd.read_csv('/content/iris.data',names = cols)
array = data.values
X = array[:,0:4]
y = array[:,4]
enc = LabelEncoder()
y_1 = y.apply(enc.fit_transform)
y_1.head()
