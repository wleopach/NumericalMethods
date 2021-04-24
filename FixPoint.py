'''
Created on 18/05/2020

@author: leo1p
'''
import numpy as np
import pandas as pd
from math import sqrt


def g1(x):
    return  x-(x**3) - 4*(x**2) +10

def g2(x):
    return  sqrt((10/x) - (4*x)) 

def g3(x):  
    return  (0.5)*sqrt((10)- (x**3))

def g4(x): 
    return sqrt((10/(4+x)))
def g5(x):
    return x- ((x**3) + 4*(x**2) -10)/((3*x**2)+8*x)
R=dict()
R[g1]=[]
R[g2]=[]
R[g3]=[]
R[g4]=[]
R[g5]=[]

p=1.5
N=30
i=0
for j in [g1, g2, g3, g4,g5]:
    q=p
    i=0 
    while i<N:
        if j(q)==q: 
            print( j.__name__ +"$--->$ iteration " + str(i) + '   valor' + str(q) +'  es un punto fijo \\\\')
            R[j].append(float(q))
            i=N
        elif  j(q)!=q and  abs(j(q)-p)>0.5: 
            
            print( j.__name__ +"$--->$iteration " + str(i) +"   valor"+ str(j(q))+'  Fuera de tolerancia\\\\')
            R[j].append(float(j(q)))
            i=N
        elif j(q)!=q and  abs(j(q)-p)<=0.5: 
            q=j(q)
            print(j.__name__ +"$--->$ iteration  "+  str(i) +"   valor" + str(q) +'\\\\')
            R[j].append(float(q))

        i=i+1
pd.set_option("display.precision", 16)
Res1=pd.DataFrame(data=R[g1])
Res2=pd.DataFrame(data=R[g2])
Res3=pd.DataFrame(data=R[g3])
Res4=pd.DataFrame(data=R[g4])
Res5=pd.DataFrame(data=R[g5])
j=0
for i in [Res1,Res2,Res3,Res4,Res5]:
    j=j+1
    print(i) 
    i.name= 'Resultado' +str(j) 
    i.to_csv('D:/ElipseWE/NumericalMethods/'+i.name+'.csv')    
    