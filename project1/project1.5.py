import numpy as np
import matplotlib.pyplot as plt

def nf(xm, mean1, v):
    p = np.exp(-((xm - mean1)**2)/(2*v))/(np.sqrt(2*np.pi*v))
    return p
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
vars = np.var(mn,axis=0)#方差
data = a1
x = np.arange(-200, 200)
y = nf(x, mean[0], vars[0])
plt.plot(x, y)
plt.hist(data, bins=300, rwidth=0.8, density=True)
plt.show()