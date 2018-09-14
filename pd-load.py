import pandas as pd
import csv

with open("<filename>.csv",'r') as f:
  with open("update_file.csv",'w') as f1:
    f.next()
    f.next()
    for line in f:
      f1.write(line.replace('\x00',''))
df = pd.read_csv('update_file.csv')
print(df)
