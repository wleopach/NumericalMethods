import numpy as np
a = 8
x = np.array([3, 5, 7, 9])
y = np.array([a, a + 1/3, a+3, a+4])
den = 0
num = 0
for i in range(len(x)):
    num = num + (x[i] - 5.3)*y[i]
    den = den + (x[i] - 5.3)**2

k = num/den
print(k)

error = 0
for i in range(len(x)):
    error = error + (y[i] - k*(x[i]-5.3))**2
print(error)