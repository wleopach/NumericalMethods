'''
Created on May 21, 2020

@author: leo1p
'''
import matplotlib.pyplot as plt
import numpy as np
import math as m
# Create the vectors X and Y
from matplotlib.patches import Polygon

def build_cartesian_plane(max_quadrant_range):
    """ The quadrant range controls the range of the quadrants"""
    l = []
    zeros = []
    plt.grid(True, color='b', zorder=0)
    ax = plt.axes()
    head_width = float(0.05) * max_quadrant_range
    head_length = float(0.1) * max_quadrant_range
    ax.arrow(0, 0, max_quadrant_range, 0, head_width=head_width, head_length=head_length, fc='k', ec='k',zorder=100, label='$x$')    
    ax.arrow(0, 0, -max_quadrant_range, 0, head_width=head_width, head_length=head_length, fc='k', ec='k', zorder=100, label='$x$')
    ax.arrow(0, 0, 0, max_quadrant_range, head_width=head_width, head_length=head_length, fc='k', ec='k', zorder=100, label='$x$')
    ax.arrow(0, 0, 0, -max_quadrant_range, head_width=head_width, head_length=head_length, fc='k', ec='k', zorder=100, label='$x$')
    counter_dash_width = max_quadrant_range * 0.02
    dividers = [0,.1,.2,.3,.4, .5, .6, .7, .8, .9, 1]
    for i in dividers:
        plt.plot([-counter_dash_width, counter_dash_width], [i*max_quadrant_range, i*max_quadrant_range], color='k')
        plt.plot([i * max_quadrant_range, i*max_quadrant_range], [-counter_dash_width, counter_dash_width], color='k')
        plt.plot([-counter_dash_width, counter_dash_width], [-i * max_quadrant_range, -i * max_quadrant_range], color='k')
        plt.plot([-i * max_quadrant_range, -i * max_quadrant_range], [-counter_dash_width, counter_dash_width], color='k')
        l.append(i * max_quadrant_range)
        l.append(-i * max_quadrant_range)
        zeros.append(0)
        zeros.append(0)


    cuad=plt.axes()
    cuad.text(max_quadrant_range, -max_quadrant_range * 0.1, '$x$')
    #cuad.text(max_quadrant_range/4, max_quadrant_range /2, 'PRIMER CAUDRANTE')
    #cuad.text(max_quadrant_range/4,-max_quadrant_range /2, 'CUARTO CAUDRANTE')
    #cuad.text(-7*max_quadrant_range/8,max_quadrant_range /2, 'SEGUNDO CAUDRANTE')
    #cuad.text(-7*max_quadrant_range/8,-max_quadrant_range /2, 'TERCER CAUDRANTE')
    cuad.text(-max_quadrant_range * 0.1,max_quadrant_range, '$y$')
    x=np.linspace(-max_quadrant_range, max_quadrant_range)
    y=np.ones(50)
    w=np.linspace(-5, 5)
    z=w**2+3  
    r=w**2 
    s=w**2-3                          
    ax.plot(w,r, label='$x^2$')
    ax.plot(w,s,label= '$x^2-3$')
    ax.plot(w, z, color='y',label= '$x^2+3$')
    
build_cartesian_plane(20)
plt.legend()
plt.show()