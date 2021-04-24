'''
Created on Jul 10, 2020

@author: leo1p
'''
import numpy as np
import pandas as pd

#def p(x):
#    return 2*x**4 -3*x**2 +3*x-4
def p(x):
    return (x**3) + 4*(x**2) -10
x=1.5
#c=[2,0,-3,3,-4]
c=[1,4,0,-10]
l=len(c)
j=0
N=20
while j<N:
    b=[]
    b.append(c[0])
    for i in range(1,l):
        d= c[i]+b[i-1]*(x)
        b.append(d)
    f=[]
    f.append(b[0])
    for i in range(1,l-1):    
        d= b[i]+f[i-1]*(x)
        f.append(d)
    w=x-(b[l-1]/float(f[l-2]))
    x=w
    j=j+1
    print(p(x),x)
