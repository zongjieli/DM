import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')#打开文件
data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]#用for in 循环以逗号将数据分隔开，并转换数据类型
mn = np.array(data)

vars = np.var(mn,axis=0)#方差
a = np.where(vars == np.max(vars))[0][0]#最大方差的位置
b = np.where(vars == np.min(vars))[0][0]#最小方差的位置
print(vars)
print(a)
print(b)