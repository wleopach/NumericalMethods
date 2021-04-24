'''
Created on Sep 28, 2020

@author: leo1p
'''
import matplotlib as mpl
# mpl.use('TkAgg') # <-- THIS MAKES IT FAST!
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.widgets as wdg
import numpy as np
import subprocess as sbp

# Do not use rc('text', usetex=True) becouse it  slows the graphics
from matplotlib import rc
#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('font', **{'family': 'serif', 'serif': ['Times']})
params = {'font.size': 11,
          # 'text.usetex': True,
          'toolbar': 'toolbar2',  # None |  toolbar2
          }
plt.rcParams.update(params)

# Direct input - error un windows
# plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
# # Options
# params = {'text.usetex': True,
#           'font.size': 11,
#           'font.family': 'fontext',
#           # 'font.family': 'serif',
#           # 'font.serif': 'Times',
#           'text.latex.unicode': True,
#           }

# plt.rcParams.update(params)  # - error un windows


## graphics class edit
class BGraph():
    def __init__(self, m=[], n=[], fig=[], axs=[],
                 Dxl=0.05, Dxr=0.05, Dx=0.03,
                 Dyu=0.05, Dyb=0.1, Dy=0.03, conf=[], shx=True, shy=False,
                 vec=1, draw=True):
        self.m = m  # number of rows
        self.n = n  # number of columns
        self.fig = fig  # figure
        self.axs = axs  # axes array in the old version m_axes
        self.Dxl = Dxl  # distance of the left axes to the figure border
        self.Dxr = Dxr  # distance of the right axes to the figure border
        self.Dx = Dx  # Distance in x between axes
        self.Dyu = Dyu  # Disran
        self.Dyb = Dyb
        self.Dy = Dy
        self.conf = conf
        self.shx = shx
        self.shy = shy
        self.vec = vec
        self.draw = draw
        self.subplots_mod()
        self.curs_l = []
        self.pick_text = []
        self.pick_mark = []
        plt.show()

    def cursor(self, i_c):
        self.curs_l.append(wdg.Cursor(self.axs[i_c], useblit=True,
                                      color='red', linewidth=1))

## label_xy
# obj is the reference object 'axs' of 'fig'
# Center te text box tx in the figure fig for the axes made with
# the subplots_mod function. Where center can be  'x', 'y' to center in
# x or y direction of 'xy' to center in both
    def label_xy(self, obj, text, center, Dxl=0.05, Dxlpos=.04, Dxr=0.05,
                 Dxrpos=1.0/4.0, Dyu=0.05, Dyb=0.1, Dybpos=1.0/4.0,
                 Dyupos=1.0/4.0,
                 color='k'):
        tx = ''
        if obj[:3] == 'axs':
            n = np.int(obj[3:])
            obj = obj[:3]
        if center == 'x':
            # if type(obj).__name__ == 'Figure':
            if obj == 'fig':
                tx = self.fig.text(0.5, 0.02, text, color=color, fontsize=20,
                                   ha='center', va='center')
                xp = Dxl + (1. - Dxl - Dxr) / 2.
                yp = Dyb * Dybpos
                tx.set_position((xp, yp))
                # self.fig.canvas.draw()
            else:
                axpos = self.axs[n].get_position()
                # fig = self.axs[n].get_figure()
                tx = self.fig.text(0.02, 0.5, text, color=color, fontsize=20,
                                   ha='center', va='center')
                xp = (axpos.x1 + axpos.x0) / 2.0
                yp = axpos.y0 - Dybpos
                tx.set_position((xp, yp))
                # self.fig.canvas.draw()
        elif center == 'xt':
            if obj == 'fig':
                # if type(obj).__name__ == 'Figure':
                tx = self.fig.text(0.5, 0.02, text, color=color, fontsize=20,
                                   ha='center', va='center')
                xp = Dxl + (1 - Dxl - Dxr) / 2.0
                yp = 1 - Dyb * Dyupos
                tx.set_position((xp, yp))
                # self.fig.canvas.draw()
            else:
                axpos = self.axs[n].get_position()
                # fig = obj.get_figure()
                tx = self.fig.text(0.02, 0.5, text, color=color, fontsize=20,
                                   ha='center', va='center')
                xp = (axpos.x1 + axpos.x0) / 2.0
                yp = 1 - Dyupos
                tx.set_position((xp, yp))
                # self.fig.canvas.draw()
        elif center == 'y':
            # if type(obj).__name__ == 'Figure':
            if obj == 'fig':
                tx = self.fig.text(0.02, 0.5, text, color=color, fontsize=20,
                                   ha='center', va='center',
                                   rotation='vertical')
                xp = Dxl * Dxlpos
                yp = Dyb + (1. - Dyu - Dyb) / 2.0
                tx.set_position((xp, yp))
            else:
                axpos = self.axs[n].get_position()
                # fig = obj.get_figure()
                tx = self.fig.text(0.02, 0.5, text, color=color, fontsize=20,
                                   ha='center', va='center',
                                   rotation='vertical')
                xp = axpos.x0 - Dxlpos
                yp = (axpos.y1 + axpos.y0) / 2.0
                tx.set_position((xp, yp))
                # self.fig.canvas.draw()
        elif center == 'yr':
            # if type(obj).__name__ == 'Figure':
            if obj == 'fig':
                tx = self.fig.text(0.02, 0.5, text, color=color, fontsize=20,
                                   ha='center', va='center',
                                   rotation='vertical')
                xp = 1.0 - Dxr * Dxrpos
                yp = Dyb + (1 - Dyu - Dyb) / 2.0
                tx.set_position((xp, yp))
                # obj.canvas.draw()
            else:
                axpos = self.axs[n].get_position()
                # fig = obj.get_figure()
                tx = self.fig.text(0.02, 0.5, text, color=color, fontsize=20,
                                   ha='center', va='center',
                                   rotation='vertical')
                xp = 1.0 - Dxrpos
                yp = (axpos.y1 + axpos.y0) / 2.0
                tx.set_position((xp, yp))
                # fig.canvas.draw()
        self.fig.canvas.draw()
        return tx

#########################################################
## #################### def_leg ############################
#########################################################
# Create a legend with markers mark, legends leg and position pos_leg

    def def_leg(self, ax, leg, mrk, pos_leg):
        h, l = self.axs[ax].get_legend_handles_labels()
        h = []
        l = []
        for i in np.arange(np.size(leg)):
            h.append(self.axs[ax].plot([], [], mrk[i])[0])
            l.append(leg[i])
            h_leg = self.axs[ax].legendg(h, l, bbox_to_anchor=pos_leg)
        return h_leg


## axs_ticks
# Modify the axes labels and ticks
# ax: axes where the axis live
# axis: axis 'x' or 'y'
# ticks: array with the ticks
# ti:
    def axs_ticks(self, ax, axis, ticks=[], tl=True):
        if axis == 'y':
            axy = self.axs[ax].get_yaxis()
            axy.set_ticks(ticks)
            axy.set_ticklabels(ticks)
            if not tl:
                axy.set_ticklabels([])
        else:
            axx = self.axs[ax].get_xaxis()
            axx.set_ticks(ticks)
            if not tl:
                axx.set_ticklabels(ticks)

## set_figure
# fig: figure to modify
# l_width: width of the lines
# m_size: size of the markers
# ax_size: size of font for axis
# tx_size: size of general text font
# lg_size: size of the legends font
    def set_figure(self, l_width=2, m_size=10, ax_size=18, tx_size=22,
                   lg_size=18, xpad=[]):
        if np.shape(xpad)[0] == 0:
            xpad = ax_size / 2.5

        for o in self.fig.findobj():
            if hasattr(o, 'set_linewidth'):
                o.set_linewidth(l_width)
            if hasattr(o, 'set_markersize'):
                o.set_markersize(m_size)
        axesf = self.axs
        for ax in axesf:
            ax.tick_params(axis='both', which='both', labelsize=ax_size)
            ax.tick_params(axis='x', which='major', pad=xpad)
            leg = ax.get_legend()
            if leg is not None:
                legt = leg.get_texts()
                for tx in legt:
                    tx.set_fontsize(lg_size)

        figt = self.fig.texts
        for tx in figt:
            tx.set_fontsize(tx_size)
        self.fig.canvas.draw()

## plotyy
    def plotyy(self, ax, xdata, y1data, y2data, y1name=r'', y2name=r'',
               f1='sr', a1=1., f2='sb', a2=1., label_1='', label_2=''):
        # fig = ax.get_figure()
        axp = self.axs[ax].get_position()
        self.axs[ax].plot(xdata, y1data, f1, alpha=a1, label=label_1)
        self.label_xy(self.fig, y1name, 'y',
                      Dxl=axp.x0, Dyb=axp.y0, color=f1[1])
        for tl in self.axs[ax].get_yticklabels():
            tl.set_color(f1[1])
        ax2 = self.axs[ax].twinx()
        l2, = ax2.plot(xdata, y2data, f2, alpha=a2)

        h, l = self.axs[ax].get_legend_handles_labels()
        h.append(l2)
        l.append(label_2)
        # print(h)
        # print(l)
        h, l = self.axs[ax].get_legend_handles_labels()
        # print('Despues')
        # print(h)
        # print(l)
        self.label_xy(self.fig, y2name, 'yr',
                      Dxr=1-axp.x1, Dyb=axp.y0, color=f2[1])
        for tl in ax2.get_yticklabels():
            tl.set_color(f2[1])
        ax2.set_position(axp)
        return ax2


## Ajusta las margenes y los espacios entre los subplots
#
# |Dxl|___________|_Dx_|________|Dxr|___
# |    __________      _________   |____Dyu
# |   |          |    |         |  |
# |   |   1,1    |    |   1,n   |  |
# |   |__________|    |_________|  |__
# |    __________      _________   |__Dy
# |   |          |    |         |  |
# |   |   m,1    |    |   m,n   |  |
# |   |__________|    |_________|  |____
# |________________________________|____Dyb
# conf: matrix that indicates de dimensions of the axes, in the way
#          [[y0_1, x0_1, L_1, H_1], ..., [x0_n, y0_n, L_n, H_n]]
#          where x0_, y0_n indicates the origin of the nth axes,  L_n indicates
#          the number of cells of x length and H_n  the numer of cells
#          of y lenght.
# shx: True to share the x axis
# sxy: True to share the y axis
# vec: 0: no vector form
#        1; arange in theta next way
#            1,     2m,    ...,     mn
#             :          :                :
#            m-1, 2m-1, ...     mn-1
#            m,    m+1, ..., n(m-1)+1
#        2; arange in theta next way
#            1,  m+1, ..., n(m-1)+1
#             :       :              :
#            2,  2m-1, ...,  mn-1
#            m, 2m,    ...,   mn
#        3: arange in the next way
#            1,     2,     ...k
#            k+1, k+2, ...l
# draw: if True shw theta figure

    def subplots_mod(self):
        if self.draw is True:
            plt.ion()
            # fig.canvas.draw()
            # fig.show()
        else:
            plt.ioff()
        if isinstance(self.fig, list):
            # if not is given a figure
            if np.shape(self.conf)[0] != 0:
                k = np.shape(self.conf)[0]
                self.fig, self.axs = plt.subplots(k, 1,
                                                  sharex=self.shx,
                                                  sharey=self.shy)
                if k == 1:
                    self.axs = [self.axs]
            # If a figure is givenx
            else:
                self.fig, self.axs = plt.subplots(self.m, self.n,
                                                  sharex=self.shx,
                                                  sharey=self.shy)
                self.axs.shape = [self.m, self.n]
        else:  # A given fig and axs
            self.m = np.shape(self.axs)[0]  # numero de filas
            self.n = np.shape(self.axs)[1]  # numero de columnas
            self.axs.shape = [self.m, self.n]

        if self.m == 1 and self.n == 1:
            self.axs = np.array([self.axs])
            self.axs.shape = [self.m, self.n]
        # Espacio disponible en x
        Lx = 1 - (self.n - 1)*self.Dx - self.Dxl - self.Dxr
        # Espacio disponible en y
        Ly = 1 - (self.m - 1)*self.Dy - self.Dyb - self.Dyu
        l = Lx/self.n
        h = Ly/self.m
        if np.size(self.conf) != 0:
            Lxp = 1 - self.Dxl - self.Dxr
            Lyp = 1 - self.Dyb - self.Dyu
            lp = Lxp/self.n
            hp = Lyp/self.m
            k = 0
            for axe_i in self.conf:
                x0 = self.Dxl + (l+self.Dx)*axe_i[1]
                if axe_i[2] == self.n:
                    x1 = x0 + axe_i[2]*lp
                else:
                    x1 = x0 + (axe_i[2]-1)*lp + l
                y0 = 1 - self.Dyu - (axe_i[0])*(self.Dy + h) - h
                if axe_i[3] == self.m:
                    y1 = y0 + axe_i[3]*hp
                else:
                    y1 = y0 + (axe_i[3]-1)*hp + h
                p = np.array([[x0, y0],
                              [x1, y1]])
                axpos = self.axs[k].get_position()
                axpos.set_points(p)
                self.axs[k].set_position(axpos)
                k = k+1
        else:  # Given m and n
            for i in range(self.m):
                for j in range(self.n):
                    x0 = self.Dxl + (l+self.Dx)*j
                    x1 = x0 + l
                    y0 = 1 - self.Dyu - i*(self.Dy + h) - h
                    y1 = y0 + h
                    p = np.array([[x0, y0],
                                  [x1, y1]])
                    axpos = self.axs[i, j].get_position()
                    axpos.set_points(p)
                    self.axs[i, j].set_position(axpos)
            if self.vec == 1:
                self.vec = np.array([])
                for j in np.arange(self.axs.shape[1]):
                    if np.power(-1, j) > 0:
                        for i in np.arange(self.axs.shape[0]):
                            self.vec = np.hstack((self.vec, self.axs[i, j]))
                    else:
                        for i in np.arange(self.axs.shape[0]-1, -1, -1):
                            self.vec = np.hstack((self.vec, self.axs[i, j]))
                self.axs = self.vec
            elif self.vec == 2:
                self.vec = np.array([])
                for j in np.arange(self.axs.shape[1]):
                    if np.power(-1, j) > 0:
                        for i in np.arange(self.axs.shape[0]):
                            self.vec = np.hstack((self.vec, self.axs[i, j]))
                    else:
                        for i in np.arange(self.axs.shape[0]):
                            self.vec = np.hstack((self.vec, self.axs[i, j]))
                self.axs = self.vec
            elif self.vec == 3:
                self.vec = np.array([])
                for axs in self.axs:
                    self.vec = np.hstack((self.vec, axs))
                self.axs = self.vec

##########################################################
##                 Create a new axis to zoom grapics                          #
##########################################################
# pt: vector with the position of the zoom axes
#      pt[0]: if 'cx' center in x-axis, 'cy' center in y-axis
#      pt[1]: y position of the zoom axes
#      pt[2]: width of the zoom axes
#      pt[3]: heigh of the zoom axes
# limits: region to zoom
#            limits[0:1]: xmin and xmax
#            limits[2:3]: ymi and ymax
# cNorm: if True take the coordinates in nomlizeed way 0.0-1.0
#              in the axes of originl data
    def plot_zoom(self, pt=[], limits=[], cNorm=False, xticks=True,
                  yticks=False):
        if cNorm is True:
            # print cNorm
            xlim = self.axs[0].get_xlim()
            ylim = self.axs[0].get_ylim()
            pt[1] = pt[1]*(ylim[1]-ylim[0]) + ylim[0]
            pt[2] = pt[2]*(xlim[1]-xlim[0])
            pt[3] = pt[3]*(ylim[1]-ylim[0])
        if pt[0] == 'cx':
            xo = limits[0] + (limits[1] - limits[0] -
                              pt[2])/2.0  # Center the ax_zoom
        pt = np.array([[xo, pt[1]], [xo+pt[2], pt[1]+pt[3]]])
        ptv = self.axs[0].transData.transform(pt)  # axs->scr
        inv = self.fig.transFigure.inverted()  # scr->fig
        ptt = inv.transform(ptv)
        self.ax_zoom = self.fig.add_axes([ptt[0, 0], ptt[0, 1],
                                          ptt[1, 0]-ptt[0, 0],
                                          ptt[1, 1]-ptt[0, 1]])
        self.ax_zoom.set_xlim([limits[0], limits[1]])
        self.ax_zoom.set_ylim([limits[2], limits[3]])

        self.axs[0].add_patch(pat.Rectangle((limits[0], limits[2]),
                                            limits[1]-limits[0],
                                            limits[3]-limits[2],
                                            fill=False))
        ax_zp = self.ax_zoom.get_position()
        ax_zpos = [[ax_zp.x0, ax_zp.y0], [ax_zp.x1, ax_zp.y1]]
        ax_zposv = self.fig.transFigure.transform(ax_zpos)  # fig->scr
        inv = self.axs[0].transData.inverted()  # scr->axs
        ax_zpost = inv.transform(ax_zposv)
        self.axs[0].plot([limits[0], ax_zpost[0, 0]],
                         [limits[3], ax_zpost[0, 1]],
                         '--k')
        self.axs[0].plot([limits[1], ax_zpost[1, 0]],
                         [limits[3], ax_zpost[0, 1]],
                         '--k')
        plt.xticks(visible=xticks)
        plt.yticks(visible=yticks)
        graficas = self.axs[0].get_lines()
        for i in graficas:
            self.ax_zoom.plot(i.get_xdata(), i.get_ydata(),
                              linestyle=i.get_linestyle(),
                              color=i.get_color(), marker=i.get_marker(),
                              markevery=i.get_markevery(),
                              dash_joinstyle=i.get_dash_joinstyle(),
                              dash_capstyle=i.get_dash_capstyle(),
                              alpha=i.get_alpha())

############################################
##                  Create a zoom in anoter axes                          #
############################################
# ax_original: index of original axes
# ax_zoom: index of zomm axes
# limits: region to zoom
#            limits[0:1]: xmin and xmax
#            limits[2:3]: ymi and ymax
# cNorm: if True take the coordinates in nomlizeed way 0.0-1.0
#              in the axes of originl data

    def plot_zoom_inf(self, ax_original, ax_zoom, limits, ec='b', fc='b'):
        # fig = ax_original.get_figure()
        lim = np.array([[limits[0], 0.0], [limits[1], 0.0]])
        ax_zp = self.axs[ax_zoom].get_position()
        ax_op = self.axs[ax_original].get_position()
        limv = self.axs[ax_original].transData.transform(lim)  # axs->scr
        inv = self.fig.transFigure.inverted()  # scr->fig
        limt = inv.transform(limv)
        ll = matplotlib.lines.Line2D([ax_zp.x0, limt[0][0]], [ax_zp.y1,
                                                              ax_op.y0],
                                     transform=self.fig.transFigure,
                                     figure=self.fig,
                                     color='k')
        lr = matplotlib.lines.Line2D([limt[1][0], ax_zp.x1], [ax_op.y0,
                                                              ax_zp.y1],
                                     transform=self.fig.transFigure,
                                     figure=self.fig,
                                     color='k')
        self.fig.lines.extend([ll, lr])
        graficas = self.axs[ax_original].get_lines()
        for i in graficas:
            self.axs[ax_zoom].plot(i.get_xdata(), i.get_ydata(),
                                   linestyle=i.get_linestyle(),
                                   color=i.get_color(),
                                   marker=i.get_marker(),
                                   markevery=i.get_markevery(),
                                   dash_joinstyle=i.get_dash_joinstyle(),
                                   dash_capstyle=i.get_dash_capstyle(),
                                   alpha=i.get_alpha())
            ax_zoom.set_xlim(limits)
        self.boxes(self.axs[ax_original],
                   [limits[0]], [limits[1]], ec=ec, fc=fc)

    ## boxes
    def boxes(self, ax, xini=[], xend=[], ymin=[], ymax=[], ec='b',
              fc='b', alpha=0.2):
        box_list = []
        ylim = self.axs[ax].get_ylim()
        xlim = self.axs[ax].get_xlim()
        if np.size(xini) == 0:
            xini = np.ones(np.size(ymin))*xlim[0]
        if np.size(xend) == 0:
            xend = np.ones(np.size(ymin))*xlim[1]
        if np.size(ymin) == 0:
            ymin = np.ones(np.size(xini))*ylim[0]
        if np.size(ymax) == 0:
            ymax = np.ones(np.size(xini))*ylim[1]
        for xin, xen, ymi, yma in np.nditer([xini, xend, ymin, ymax]):
            box_list.append(
                self.axs[ax].add_patch(pat.Rectangle((xin, ymi),
                                                     xen-xin,
                                                     yma-ymi,
                                                     ec=ec,
                                                     fc=fc,
                                                     fill=True,
                                                     alpha=alpha)))

    ## Activate on click retrieve data
    def activ_press(self, hab_button_press=True):
        self.hab_button_press = hab_button_press
        if hab_button_press:
            self.cn_button_press = self.fig.canvas.mpl_connect(
                'button_press_event',
                self.button_press)
        else:
            self.fig.canvas.mpl_disconnect(self.cn_button_press)

    def button_press(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        print('(x, y) = (%f, %0.2f)' % (x, y))

    ## Activate  click movement retrieve data
    def activ_move(self, hab_button_move=True):
        self.hab_button_move = hab_button_move
        if hab_button_move:
            self.cn_button_press = self.fig.canvas.mpl_connect(
                "motion_notify_event",
                self.on_move)
        else:
            self.fig.canvas.mpl_disconnect(self.cn_button_press)

    def on_move(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        print('(x, y) = (%f, %0.2f)' % (x, y))

    ## Activate pick element to retrieve data
    def activ_pick(self, hab_pick=True):
        self.hab_pick = hab_pick
        if hab_pick:
            self.cn_button_press = self.fig.canvas.mpl_connect(
                "pick_event",
                self.on_pick)
        else:
            self.fig.canvas.mpl_disconnect(self.cn_button_press)

    def on_pick(self, event):
        self.event = event
        ax = self.event.artist.axes
        line = event.artist
        xdata, ydata = line.get_data()
        ind = event.ind[-1]
        xdata = np.array(xdata[ind])
        ydata = np.array(ydata[ind])
        if np.size(self.pick_text) != 0:
            self.pick_text.remove()
            self.pick_mark[0].remove()
        bbox = dict(boxstyle="round", fc="0.8")
        offset = 7
        # self.pick_text = ax.text(xdata, ydata, r'%.2f, %.2f' % (xdata, ydata),
        #                          fontsize=11, backgroundcolor = 'y', alpha=.8,
        #                          bbox=bbox, xytext=(0.8*offset, -offset))
        arrowprops = dict(arrowstyle="->",
                          connectionstyle="angle,angleA=0,angleB=90,rad=10")
        self.pick_text = ax.annotate('(%.2f, %.2f)' % (xdata, ydata),
                                     (xdata, ydata), xytext=(3.8*offset,
                                                             -offset),
                                     # xycoords='figure pixels',
                                     textcoords='offset points',
                                     bbox=bbox, arrowprops=arrowprops)
        self.pick_text.set_backgroundcolor('y')
        self.pick_mark = ax.plot(xdata, ydata, 'sr')
        self.fig.canvas.draw()


 ## close all windows   
def close_all():
    for i in plt.get_fignums():
        plt.close(i)


## get_screen_resolution
def get_screen_resolution():
    output = sbp.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True,
                       stdout=sbp.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return {'width': float(resolution[0]), 'height': float(resolution[1])}

## class test
# x = np.linspace(-3, 3, 100)
# y1 = x*x
# y2 = np.sin(x)

# Tst = BGraph(1, 2)
# Tst.axs[0].plot(x, y1, picker=5)
# Tst.axs[1].plot(x, y2, picker=5)
# Tst.cursor(1)
# Tst.cursor(0)
# Tst.activ_pick()
