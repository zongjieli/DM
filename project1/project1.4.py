import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')#打开文件
data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]#用for in 循环以逗号将数据分隔开，并转换数据类型
mn = np.array(data)
n = mn.shape[0]#得到矩阵的行数
mean = np.mean(mn, axis=0)
cm = mn - np.ones([n, 1]) * mean#中心化矩阵

a1 = cm[:, 0]
a2 = cm[:, 1]
cos = np.dot(a1, a2)/(np.dot(a1, a1) * np.dot(a2, a2))
print(cos)
plt.scatter(a1,a2)
plt.show()