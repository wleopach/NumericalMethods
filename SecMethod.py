# -*- coding: utf-8 -*-
'''
Created on 19/05/2020

@author: leo1p
'''
import numpy as np
import pandas as pd
from io import StringIO
from tabulate import tabulate
import math as m
N=10
T= 2
p= m.pi/4
r= 0.5
i=0

def f(x):
#    return (x**3) + 4*(x**2) -10
    return m.cos(x)-x
def df(x,y):
#    return (3*(x**2)) +(8*x)
    return (f(x)-f(y))/float(x-y)
def g(x,y):
    return x-(f(x)/float(df(x,y)))
def t(x,y):
    return abs(f(x)/float(df(x,y)))
R=dict()
R['Iteración']=[]
R['Resultado']=[]
q=p
h=r

while i<N:
    if  f(q) ==0: 
        print( str(q) + "es una  ra�z de f")
        R['Iteración'].append(i)
        R['Resultado'].append(str(q) + " es una  raíz de f")
        i=N+1
    elif f(q)!=0 and t(q,h)< T:
        R['Iteración'].append(i)
        R['Resultado'].append("La nueva aproximación es "+ str(q))
        i=i+1
        a=q
        q=g(q,h)
        h=a
        print("La nueva aproximación es "+ str(q))
        
    elif f(q)!=0 and t(q,h)>= T:
        R['Iteración'].append(i)
        R['Resultado'].append("Fuera de tolerancia  " + str(g(q,h)))
        print("Fuera de tolerancia  " + str(g(q,h)) )
        i=N+1
R=pd.DataFrame(data=R)
print(tabulate(R, headers='keys', tablefmt='psql'))
      