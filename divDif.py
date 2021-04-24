'''
Created on Jul 28, 2020

@author: leo1p
'''
import numpy as np
import matplotlib.pyplot as plt

def coef(x, y):
    '''x : array of data points
       y : array of f(x)  '''
    x.astype(float)
    y.astype(float)
    n = len(x)
    a = []
    for i in range(n):
        a.append(y[i])

    for j in range(1, n):

        for i in range(n-1, j-1, -1):
            a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])

    return np.array(a) # return an array of coefficient

def Eval(a, x, r):
    ''' a : array returned by function coef()
        x : array of data points
        r : the node to interpolate at  '''
    x.astype(float)
    n= len( a ) - 1
    temp = a[n] + (r - x[n])
    for i in range( n - 1, -1, -1 ):
        temp = temp * ( r - x[i] ) + a[i]
    return temp # return the y_value interpolation 

x=np.array([0,1,2,3,4,5])
y=np.array([1,2,4,8,16,32])

print(coef(x,y))
print(Eval(coef(x,y),x,10))