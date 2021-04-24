import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np

a = 8
x = np.array([1, 2, 3, 4, 5])
y = np.array([a, a+2, a+4, a+8, a+1])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, c)
ax = plt.plot(x, y, 'o', label='data original', markersize=10)
ax = plt.plot(x, m*x + c, 'r', label='Linea ajustada')
ax = plt.legend()
error = LA.norm(m*x+c -y)
print(error**2)

y1 = 0
for i in y:
    y1 = y1+i
print(y1)

y2 = 0
pos = 0
for i in y:
    y2 = y2+i*x[pos]
    pos = pos+1
print(y2)

y3 = 0
pos = 0
for i in y:
    y3 = y3+i*(x[pos]**2)
    pos = pos + 1
print(y3)

