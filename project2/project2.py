import numpy as np
import csv
import pandas as pd
data=pd.read_csv('iris.csv')
data=np.array(data)
data=np.mat(data[:,0:4])
l=len(data)
k=np.mat(np.zeros((l,l)))
for i in range(0,l):
    for j in range(i,l):
        k[i,j]=(np.dot(data[i],data[j].T))**2