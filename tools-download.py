from __future__ import division, print_function
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os 
import csv
data = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

def download_data(path):
  filename = (path.split('/')[-1])
  dir = os.path.join(os.path.abspath(os.path.dirname(filename)))
  r = requests.get(path,allow_redirects=True, stream = True,verify = False)
  if r.status_code == 200:    
    with open(dir + '/' + filename ,'wb') as f:
      f.write(r.content)
download_data(data)
print(os.path.join(filename + '.csv'))
data = pd.read_csv(filename)
