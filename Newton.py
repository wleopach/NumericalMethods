# -*- coding: utf-8 -*-
'''
Created on 18/05/2020

@author: leo1p
'''
import numpy as np
import pandas as pd
from io import StringIO
from tabulate import tabulate
import math as m
N=10
T= 10**(-5)
p= -4
i=0

def f(x):
    return (x**4) + 5*(x**3) -9*(x**2)-85*x-136
#    return m.cos(x)-x
def df(x):
    return (4*(x**3)) +(15*(x**2)) -18*x -85
#    return -(m.sin(x)+1)
def g(x):
    return x-(f(x)/float(df(x)))
def t(x,y):
    return abs(x-y)
R=dict()
R['Iteración']=[]
R['Resultado']=[]
q = p

while i < N:
    if  f(q) ==0: 
        print( str(q) + "es una  raíz de f")
        R['Iteración'].append(i)
        R['Resultado'].append(str(q) + " es una  raíz de f")
        i=N+1
    elif f(q)!=0 and t(p , g(q)) >= T:
        R['Iteración'].append(i)
        R['Resultado'].append("La nueva aproximación es "+ str(q))
        i=i+1
        p=q
        q=g(q)
        print("La nueva aproximación es "+ str(q))
        
    elif f(q)!=0 and t(p , q) < T:
        R['Iteración'].append(i)
        R['Resultado'].append("Ya hay la presicion  " + str(g(q)))
        print("Ya hay la presicion  " + str(g(q)) )
        i=N+1
R=pd.DataFrame(data=R)
print(tabulate(R, headers='keys', tablefmt='psql'))
      