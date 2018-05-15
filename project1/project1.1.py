import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')#打开文件
data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]#用for in 循环以逗号将数据分隔开，并转换数据类型
mn = np.array(data)#矩阵

mean = np.mean(mn, axis=0)#利用函数计算每一列的均值
print(mean)
