'''
Created on Aug 16, 2020

@author: leo1p
'''
def a1(x):
    return x*(x-1)

def a2(x):
    return x*((x-1)**2)/2

def a3(x):
    return x*((x-1)**3)/6
def a4(x):
    return x*((x-1)**4)/(24)


x=19
print(a1(x),a2(x),a3(x),a4(x))
print(x-a1(x)+ 2*a2(x)-6*a3(x)+24*a4(x))