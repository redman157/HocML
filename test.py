import pandas as np
import numpy as np
X_test = None
record = pd.read_csv('test.csv')
record = record.Image
for i in range(record.shape[0]):
  if X_test is None :
    X_test = np.array(record[i].split(' '),dtype = np.int32)
  else:
    X_test = np.concatenate((X_test,np.array(record[i].split(' '),dtype = np.int32)),axis = 0)
else:
  print(X_test)
