'''
Created on Sep 28, 2020

@author: leo1p
'''
import numpy as np
# opcional para tener una carpeta con las funciones personales
import sys

# Si no funciona entonces el archivo graphedit.py debe estar en
# la misma carpeta que el script qeu vamos a programar
import graphedit as gph


# Ayuda de la funcion:
# x : dato para el cual quiero calcular el valor interpolado
# x_expe : datos de x dados para la interpolacion
# y_expe : datos de y usados para la interpolacion
# return: retorna el valor interpolado para x
def lagrange_interpol(x, x_expe, y_expe):
    # orden de polinomio menos 1
    n = x_expe.size
    # vector donde guardamos los l_i
    l_i = np.ones((n))
    # llenamos el vector l_i
    for i in np.arange(n):
        for j in np.arange(n):
            if j != i:
                l_i[i] = l_i[i]*(x - x_expe[j])/(x_expe[i]-x_expe[j])
    return np.sum(y_expe*l_i)


## codigo de interpolacion usando Lagrange
# datos iniciales, experimentales o dados por el fabricante
x_expe = np.linspace(-2, 4, 10)
#Modificamos el Codigo para que los puntos de 
x_expe=np.array([-2,-1.5,0,1.5,2,3,3.1,3.5,3.75]) 
y_expe = x_expe*x_expe
y_expe[2] = y_expe[2]+0.5
# x para los cuales quiero calcular la interpolacion
x = np.linspace(-2, 4, 1000)
# creamos un arreglo vacio
P_nx = np.array([])
for i_x in x:
    P_temp = lagrange_interpol(i_x, x_expe, y_expe)
    P_nx = np.hstack((P_nx, P_temp))


# creamos la hoja para graficar
GLi = gph.BGraph(1, 1, vec=1,
                 Dx=.05, Dxl=0.12, Dxr=0.01,
                 Dy=.1, Dyu=0.01, Dyb=0.12,
                 shy=False, shx=False)

# https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.plot.html
# hacemos la gráfica de los puntos experimentales
GLi.axs[0].plot(x, P_nx, '.b', label=r'Datos interpolados')
GLi.axs[0].plot(x_expe, y_expe, 'og', label=r'Datos experimentales')
GLi.axs[0].plot(x, x*x, '-k', label=r'Datos real')

# genera las leyendas de que es cada linea
GLi.axs[0].legend()

# nombres a los ejes de la gafica 0
GLi.label_xy('axs0', r'$x$', 'x', Dybpos=0.09)
GLi.label_xy('axs0', r'$f(x)$', 'y', Dxlpos=0.09)
# Ajustar los tamanos de la grfica
GLi.set_figure(l_width=3, ax_size=30, tx_size=35, lg_size=40,
               m_size=10)
GLi.fig.set_size_inches(15, 15)

