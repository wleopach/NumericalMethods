# -*- coding: utf-8 -*-
'''
Created on Jun 11, 2020

@author: leo1p
'''
import numpy as np
import pandas as pd
from io import StringIO
from tabulate import tabulate
import math as m

#PARAMS
N=50
a=1
b=2   
i=0
def f(x):
     return  (x**3) + 4*(x**2) -10

R=dict()
R['Iteracion']=[]
R['Resultado']=[]
r=a
s=b
p=(a+b)/2
while i<N:
    
    if f(p)==0:
        print( str(p) + "es una  raíz de f")
        R['Iteracion'].append(i)
        R['Resultado'].append(str(p) + " es una  raíz de f")
        i=N
    elif f(r)*f(p)>0:
        i=i+1
        R['Iteracion'].append(i)
        R['Resultado'].append(str(p) + " es la nueva aproximación")
        r=p
        p=(r+s)/2
    elif f(s)*f(p)>0:
        i=i+1
        R['Iteracion'].append(i)
        R['Resultado'].append(str(p) + " es la nueva aproximación")
        s=p
        p=(r+s)/2
        
R=pd.DataFrame(data=R)
print(tabulate(R, headers='keys', tablefmt='psql'))
             
        
    
    

