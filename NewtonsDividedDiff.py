# -*- coding: utf-8 -*-
"""

@author: leo1p
"""
import numpy as np

li = [0, 1, 2, 3, 4]
a = 19
y = [0, 2 * a, 4 * a, 6 * a, 8 * a]


def f(x):
    return np.cos(x)


s = (len(li), len(li))

F = np.zeros(s)

it = range(len(li))

for i in it:
    #F[0, i] = f(li[i])
    F[0, i] = y[i]
r = 1
for i in it:
    for j in it:
        if j >= i >= r:
            F[i, j] = (F[i-1, j] - F[i-1, j-1]) / (li[j] - li[j - i])

r = r+1
for i in it:
    print(F[i, i])

ap = F[0, 0] + F[1, 1]*1.5 + F[2, 2]*1.5*(1.5-1) + F[3, 3]*1.5*(1.5-1)*(1.5-2) + F[4, 4]*1.5*(1.5-1)*(1.5-2)*(1.5-3)

print(ap)