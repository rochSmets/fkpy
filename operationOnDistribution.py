
import numpy as np
from scipy import integrate
from pylab import meshgrid
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.cm as cm
from matplotlib import colorbar



class Populations():

    def __init__(self):
        self.populations = []


    def add(self, n=1.0, bulk=[0.0, 0.0], vth=[0.5, 0.5], support=[[-np.inf, +np.inf], [-np.inf, +np.inf]]):
        idPop = len(self.populations)+1
        params = {'idPop' : idPop, 'n' : n, 'bulk' : bulk, 'vth' :  vth, 'support' : support}
        self.populations.append(params)


    def cut(self, pop_list = 'All', support = [[0.0, +np.inf], [0.0, +np.inf]]):
        if pop_list == 'All':
            pop_list = [*range(len(self.populations))]
        elif type(pop_list) is int:
            pop_list = [pop_list]

        for i in pop_list:
            self.populations[i]['support'] = support


    def show(self, pop_list = 'All'):
        if pop_list == 'All':
            pop_list = [*range(len(self.populations))]
        elif type(pop_list) is int:
            pop_list = [pop_list]

        for i in pop_list:
            print('Population Id    {}'.format(self.populations[i]['idPop']))
            print('Density          {}'.format(self.populations[i]['n']))
            print('Bulk velocity    {}'.format(self.populations[i]['bulk']))
            print('Thermal velocity {}'.format(self.populations[i]['vth']))
            print('Support          {}'.format(self.populations[i]['support']))


    def computeDistrib(self):
        pop_list = [*range(len(self.populations))]

        def func_(x, y):
            f_ = 0
            for i in pop_list:
                n_ = self.populations[i]['n']
                bulk_ = self.populations[i]['bulk']
                vth_ = self.populations[i]['vth']
                cut_ = self.populations[i]['support']

                if cut_[0][0] != -np.inf and cut_[0][1] != -np.inf:
                    fx_ = lambda x : np.heaviside(x-cut_[0][0], 0)*np.heaviside(cut_[0][1]-x, 0)
                elif cut_[0][0] != -np.inf and cut_[0][1] == -np.inf:
                    fx_ = lambda x : np.heaviside(x-cut[0][0], 0)
                elif cut_[0][0] == -np.inf and cut_[0][1] != -np.inf:
                    fx_ = lambda x : np.heaviside(cut_[0][1]-x, 0)
                else:
                    fx_ = lambda x : 1

                if cut_[1][0] != -np.inf and cut_[1][1] != -np.inf:
                    fy_ = lambda y : np.heaviside(y-cut_[1][0], 0)*np.heaviside(cut_[1][1]-y, 0)
                elif cut_[1][0] != -np.inf and cut_[1][1] == -np.inf:
                    fy_ = lambda y : np.heaviside(y-cut_[0][0], 0)
                elif cut_[1][0] == -np.inf and cut_[1][1] != -np.inf:
                    fy_ = lambda y : np.heaviside(cut_[1][1]-y, 0)
                else:
                    fy_ = lambda y : 1

                norm_ = n_/(2*np.pi*vth_[0]*vth_[1])
                f_ += norm_*np.exp(-0.5*((x-bulk_[0])/vth_[0])**2)*np.exp(-0.5*((y-bulk_[1])/vth_[1])**2)*fx_(x)*fy_(y)

            return f_
        self.distrib = func_
        #self.distrib = lambda x, y: sum([f(x, y) for f in myFuncs])


    def display(self, pop_list='All', domain=[-3.0, +3.0, -3.0, +3.0], nbins=[80, 80], bounds=[-6.0, 0.0]):
        xx = np.linspace(domain[0], domain[1], nbins[0])
        yy = np.linspace(domain[2], domain[3], nbins[1])
        xp, yp = meshgrid(xx, yy)

        val = self.distrib(xp, yp)
        val = np.clip(val, np.finfo(np.float).eps, np.inf)

        zz = np.log(val)
        #zz = np.add(zz, -np.max(zz))
        #zz = np.clip(zz, bounds[0], bounds[1])

        fig = plt.figure(num=0, figsize=[6.0, 5.0], dpi=100)
        #matplotlib.figure.Figure.clear(fig)
        fig.clear(True)

        ax = fig.add_subplot(111)

        im = ax.imshow(zz,
                       aspect = 'auto',
                       interpolation = 'nearest',
                       cmap = cm.get_cmap('viridis_r', 32),
                       origin = 'lower',
                       extent = domain,
                       vmin = bounds[0],
                       vmax = bounds[1])

        # co = ax.contour(xp, yp, np.transpose(zz), 20,
        #                 colors = ('k',),
        #                 origin = 'lower',
        #                 extent = domain,
        #                 linestyles =  'solid',
        #                 linewidths = 1)

        cbar = fig.colorbar(im, ticks = np.linspace(bounds[0], bounds[1], num=3), pad = 0.03, aspect = 40)

        # plt.savefig('distrib.pdf')


    def density(self, show=True):
        n_ = integrate.dblquad(self.distrib, -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        if show == True:
            print('density (moment of order 0) : {:.4f}'. format(n_))

        return n_


    def bulk(self, show=True):
        n_ = self.density(show=False)

        wx = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), x), -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]
        wy = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), y), -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        if show == True:
            print('bulk[0] (moment of order 1) : {:.4f}'. format(wx/n_))
            print('bulk[1] (moment of order 1) : {:.4f}'. format(wy/n_))

        return [wx/n_, wy/n_]


    def pressure(self, show=True):
        w_ = self.bulk(show=False)

        pxx = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.square(x-w_[0])),\
                -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]
        pxy = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.multiply(x-w_[0], y-w_[1])),\
                -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]
        pyy = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.square(y-w_[1])),\
                -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        if show == True:
            print('pressure [0][0] (moment of order 2) : {:.4f}'. format(pxx))
            print('pressure [0][1] (moment of order 2) : {:.4f}'. format(pxy))
            print('pressure [1][1] (moment of order 2) : {:.4f}'. format(pyy))

        return [pxx, pxy, pyy]


    def heatFlux(self, show=True):
        w_ = self.bulk(show=False)

        qxxx = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.power(x-w_[0], 3)),\
                 -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        qxxy = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.multiply(np.square(x-w_[0]), y-w_[1])),\
                 -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        qxyy = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.multiply(x-w_[0], np.square(y-w_[1]))),\
                 -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        qyyy = integrate.dblquad(lambda x, y : np.multiply(self.distrib(x, y), np.power(y-w_[1], 3)),\
                 -np.inf, +np.inf, lambda x : -np.inf, lambda x : +np.inf)[0]

        if show == True:
            print('heat flux [0][0][0] (moment of order 3) : {:.4f}'. format(qxxx))
            print('heat flux [0][0][1] (moment of order 3) : {:.4f}'. format(qxxy))
            print('heat flux [0][1][1] (moment of order 3) : {:.4f}'. format(qxyy))
            print('heat flux [1][1][1] (moment of order 3) : {:.4f}'. format(qyyy))

        return [qxxx, qxxy, qxyy, qyyy]


