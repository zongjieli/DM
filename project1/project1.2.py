import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')#打开文件
data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]#用for in 循环以逗号将数据分隔开，并转换数据类型
mn = np.array(data)
n = mn.shape[0]#得到矩阵的行数
mean = np.mean(mn, axis=0)
cm = mn - np.ones([n, 1]) * mean#中心化矩阵
ccm = np.transpose(cm)#对Z进行转置
inner_product = np.dot(ccm, cm)/n#通过计算获得内积
print(inner_product)