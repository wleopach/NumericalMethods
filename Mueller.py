# -*- coding: utf-8 -*-
'''

@author: leo1p
'''
import numpy as np
import pandas as pd
from io import StringIO
from tabulate import tabulate
import math as m

N = 100
T = 10**(-5)
p0 = 0
p1 = 1
p2 = 3
i = 3

def f(x) :
    #return (x**4) + 5*(x**3) -9*(x**2)-85*x-136
     return x**3- x**2-x-1
def t(x,y) :
    return abs(x-y)

R=dict()
R['Iteración'] = []
R['Resultado'] = []

while i < N:
    h1 = p1-p0
    h2 = p2-p1
    delta1 = (f(p1)-f(p0))/float(h1)
    delta2 = (f(p2)-f(p1))/float(h2)
    d = (delta2-delta1)/(h2+h1)
    b = delta2 + h2*d
    D = np.lib.scimath.sqrt(b**2 - 4*f(p2)*d)
    if abs(b-D) < abs(b+D):
        E = b+D
    else:
        E = b-D
    h = -2*f(p2)/E
    p = p2+h
    i = i+1
    R['Iteración'].append(i)
    R['Resultado'].append("La nueva aproximación es " + str(p))
    if abs(h) < T:
        R['Iteración'].append(i)
        R['Resultado'].append("Ya hay la presicion  " + str(p))
        print("Ya hay la presicion  " + str(p))
        i = N + 1
    else:
        p0 = p1
        p1 = p2
        p2 = p

R=pd.DataFrame(data=R)
print(tabulate(R, headers='keys', tablefmt='psql'))



