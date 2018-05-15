import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')#打开文件
data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]#用for in 循环以逗号将数据分隔开，并转换数据类型
mn = np.array(data)
n1 = mn.shape[0]#得到矩阵的行数
mean = np.mean(mn, axis=0)
cm = mn - np.ones([n1, 1]) * mean#中心化矩阵
m= mn.shape[1]#列
outer_product = np.zeros([m, m])
for cmi in cm:

    cmi = np.transpose(cmi)
    ccmi = np.transpose(cmi)
    outer_product += ccmi *cmi
