import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')#打开文件
data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]#用for in 循环以逗号将数据分隔开，并转换数据类型
mn = np.array(data)
n = mn.shape[0]
m = mn.shape[1]
x=mn[0:9]
y=mn[0:m]
c=np.cov(x,y)

print(c)