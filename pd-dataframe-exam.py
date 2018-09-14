import pandas as pd
import os

new_dataframe = pd.DataFrame({
    "column_1" : [1,2,3,4,5],
    "another_column" : ['this','column','has','strings','inside'],
    "float_column": [0.1,0.5,33,48,42],
    "binary_solo": [True,False,True,True,False]
})
new_dataframe
