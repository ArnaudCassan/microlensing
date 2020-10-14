# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

import sys, os
import warnings
import numpy as np
import pandas as pd
from copy import copy
import ftplib
import subprocess
from io import StringIO
import itertools
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import scipy.optimize as opt
from scipy.interpolate import interp1d
from scipy import stats
from scipy import ndimage
import h5py
from microlensing.mismap.smallestenclosingcircle import make_circle
from sympy import Interval, Union, EmptySet
from microlensing.mismap.caustics import Caustics
from microlensing.mismap.vbb.vbb import vbbmagU
from microlensing.utils import checkandtimeit, verbosity, printi, printd, printw
from microlensing.multipoles.multipoles import quadrupole, _solvelenseq

class MagnificationCurve():
    """Magnification curve objects
        
        What it is.
        
        Includes run commands can be changed with
        rc(key, value), cf. below.
        
        :call:
        mc = MagnificationCurve(id=0)
        
        Attributes
        ----------
        params: dictionary
            ...
        
        :inputs:
        id, an integer identifying the object.
        
        :user methods:
        mc.createmc(params), create optimized magnification curve from params.
        mc.write(fmclib), write magnification curve in HDF5 library.
        mc.read(fmclib), read magnification curve from HDF5 library.
        mc.plot(), plot magnification curve.
        mc.plotraj(), plot caustics, cusps positions and trajectory.
        
        Run commands
        ------------
        'sigrc' : float
            Radius of region of influence (Gaussian sigma).
            Default is: sigrc=2.
        'caustic' : str
            Default is: caustic='central'.
        'tmin' : float
            Default is: tmin=-4.
        'tmax' : float
            Maximum Eins...
            Default is: tmin=4.
        'linerr': 0.005 ETC.ETC.
        """
    def __init__(self):
        # mc parameters and information
        self.params = dict()
        self.content = dict()
        # run command keywords
        self.rcdict = {
            'tmin': -5.,                # minimum t in tE units used to draw mc
            'tmax': 5.,                 # maximum t in tE units used to draw mc
#            'dtcroi': 6e-6,           # minimum time step in RE units inside croi
#            'Nmaxcroi': 1000000,           # maximum number of time steps inside croi
#            'extcroi': 3.,              # add length in rho units to croi radius
            'vbbacc': 0.005,            # vbb image contouring magnification accuracy
            'linerr': 0.00005             # maximum error in linear mc approximation
            }

    def rc(self, key, *value):
        """Set run commands"""
        # set I/O shell display
        tbf, tcol, tend, tit = "\033[1m", "\033[31m", "\033[0m", "\033[3m"
        # check whether run command exists
        if key not in self.rcdict:
            raise KeyError("run command keyword '" + key + "' does not exist")
        # if argument value exists, set new value, otherwise display value
        if value:
            self.rcdict[key] = value[0]
        printi(tcol + " Run command (magnification curve) " + tend + tit + key + tend + tcol + " is set to " + tend + tit + str(self.rcdict[key]) + tend)

#    def _getcroi(self):
#        """Compute caustics regions of influence (croi)"""
#        # set I/O shell display
#        tend, tcol, tit = "\033[0m", "\033[0m\033[35m", "\033[0m\033[3m"
#
#        # compute cusps positions
#        cc = Caustics(self.params['s'], self.params['q'], N=None, cusp=True)
#        self.content['topo'] = cc.topo
#        croi = dict()
#        for key in cc.cusps:
#            # create croi using smallest enclosing circle
#            points = [(np.real(p), np.imag(p)) for p in cc.cusps[key]]
#            cx, cy, r = make_circle(points)
#            croi[key] = np.array([cx, cy, r])
#        self.content['croi'] = croi
#
#        # verbose
#        print tcol + "Compute caustic regions of influence" + tit + "\n  (" + self.content['topo'] + ": s=" + str(self.params['s']) + ", q=" + str(self.params['q']) + ")" + tend

#    def _testcroi(self):
#        """Compute caustics regions of influence (croi)"""
#        # set I/O shell display
#        tend, tcol, tit = "\033[0m", "\033[0m\033[35m", "\033[0m\033[3m"
#
#        # compute cusps positions
#        cc = Caustics(self.params['s'], self.params['q'], N=None, cusp=True)
#        self.content['topo'] = cc.topo
#        croi = dict()
#        for key in cc.cusps:
#            # compute CROI radius using smallest enclosing circle
#            points = [(np.real(p), np.imag(p)) for p in cc.cusps[key]]
#            cx, cy, r = make_circle(points)
#            i = 0
#            for cx, cy in points:
#                croi[key + '_' + str(i)] = np.array([cx, cy, 0.])
#                i += 1
#
#        # verbose
#        print tcol + "Compute caustic regions of influence" + tit + "\n  (" + self.content['topo'] + ": s=" + str(self.params['s']) + ", q=" + str(self.params['q']) + ")" + tend
#
#        # save croi
#        self.content['croi'] = croi

    def create(self, params, calcmc=True):
        """Create an optimized magnification curve (mc)
            
            Parameters
            ----------
            params: dictionary
                Magnification curve input parameters.
                Keys are model parameters: s, q, u0, alpha, rho.
                
            Attributes
            ----------
            params : dictionary
                Magnification curve input parameters.
                Keys are model parameters: 's', 'q', 'u0', 'alpha' and 'rho'.
            content : dictionary
                Instance outputs. Keys are:
                'mc' : interp1d object
                    Optimized magnification curve using linear interpolation.
                'N' : int
                    Number of points in mc.
                'mumax' : float
                    Maximum magnification reached in mc.
                'croi' : dictionary of 3-elements complex ndarrays
                    Caustic regions of interest (central, secondary, resonant).
                    Fromat {key: ndarray[center_x, center_y, radius]}.
            """
        # set I/O shell display
        tit, tcol, tend = "\033[0m\033[3m", "\033[0m\033[36m", "\033[0m"
        
        # store current parameters
        self.params = params
        
        # compute global caustic influence regions
        cc = Caustics(self.params['s'], self.params['q'], N=512, cusp=True)
        
        self.content['topo'] = cc.topo

        printi(tcol + "Computing caustics and cusp regions of influence" + tend)
        printi(tit + "  ({0:s}: s={1:.4f}, q={2:.4e})".format(self.content['topo'], self.params['s'], self.params['q']) + tend)

        self.content['croi'] = dict()
       
        for key in cc.cusps:
            # use smallest enclosing circle
            points = [(np.real(p), np.imag(p)) for p in cc.cusps[key]]
            cx, cy, r = make_circle(points)
            self.content['croi'][key] = np.array([cx, cy, r])
    
        self.content['cusps'] = cc.cusps
    
#        if calcroi: self._getcroi()
#        if calcroi: self._testcroi()
#        # if croi and topo not loaded independently, raise error
#        if 'croi' not in self.content:
#            raise KeyError("magnification curve attribute content['croi'] missing")
#        if 'topo' not in self.content:
#            raise KeyError("magnification curve attribute content['topo'] missing")

        # create magnification curve
        if calcmc:
            printi(tcol + "Creating magnification curve" + tend)
            printi(tit + "  (" + self.content['topo'] + ": s={0:.4f}, q={1:.4e}, u0={2:.4e}, alpha={3:.4f}, rho={4:.4e})".format(self.params['s'], self.params['q'], self.params['u0'], self.params['alpha'], self.params['rho']) + tend)

    #        # find intersection of trajectory with croi
    #        L = list()
    #        for key in self.content['croi']:
    #            cx, cy, r = self.content['croi'][key]
    #
    #            # (u0c, t0c) of trajectory passing through croi center
    #            u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
    #            t0c =   cx * np.cos(params['alpha']) + cy * np.sin(params['alpha'])
    #
    #            # ilocal uc centered on croi
    #            uc = params['u0'] - u0c
    #
    #            # check whether the trajectory enter the extended croi
    ##            extr = r * self.rcdict['extcroi'] ## multiply radius
    #            extr = r + self.rcdict['extcroi'] * self.params['rho'] ## add radius NB : ici r est vu avec _testcroi
    #            if np.abs(uc) < extr:
    #                # time spent in extended croi
    #                Dt = np.sqrt(extr**2 - uc**2)
    #                L.append(Interval(t0c - Dt, t0c + Dt))

            # compute intervals
            self._ccroi = list()
            L = list()
            
    #        for key in self.content['croi']:
    #            cx, cy, r = self.content['croi'][key]
    #
    #            # (u0c, t0c) of trajectory passing through croi center
    #            u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
    #            t0c =   cx * np.cos(params['alpha']) + cy * np.sin(params['alpha'])
    #
    #            # ilocal uc centered on croi
    #            uc = params['u0'] - u0c
    #
    #            # check whether the trajectory enter the extended croi TEST by 9/10
    #            r = self.rcdict['extcroi'] * self.params['rho'] ** 0.9
    #            if np.abs(uc) < r:
    #                # time spent in extended croi
    #                Dt = np.sqrt(r**2 - uc**2)
    #                L.append(Interval(t0c - Dt, t0c + Dt))

    #        #    near global regions of influence
    #        t = self.rcdict['tmin']
    #        tbas = list()
    #        while t < self.rcdict['tmax']:
    #            u = np.sqrt(self.params['u0'] ** 2 + t ** 2)
    #            if u > 0.8:
    #                dt = 0.05 * u
    #            else:
    #                dt = np.max([0.007 * u, 1.e-4])
    #            t += dt
    #            tbas.append(t)
    #        tbas = np.array(tbas)
    #
    #        for key in self.content['croi']:
    #            if self.content['topo'] == 'close':
    #                for cr in [self.content['croi']['secondary_up'], self.content['croi']['secondary_down']]
    #                    cx, cy, r = cr
    #
    #                    u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
    #                    t0c =   cx * np.cos(params['alpha']) + cy * np.sin(params['alpha'])
    #
    #                    uc = params['u0'] - u0c
    #
    #                    r = r + 2. * self.rcdict['extcroi'] * self.params['rho']
    #
    ##                self._ccroi.append(np.array([cx, cy, r])) # for detailed plot
    #
    #                    if np.abs(uc) < r:
    #                        Dt = np.sqrt(r**2 - uc**2)
    #                        tex.append(tbas[(tbas <= S.right) & (tbas >=SR.left)]
    ##                        L.append(Interval(t0c - Dt, t0c + Dt))
    #
    #
    #
    #                t = tex[(tex <= S.right) & (tex >=SR.left)]
    #
    #                cx, cy, r = self.content['croi']['secondary_down']
    #
    #            if self.content['topo'] == 'wide':
    #                cx, cy, r = self.content['croi']['secondary']

    #        #    near global regions of influence
    #        if self.content['topo'] == 'close':
    #            cl = [self.content['croi']['secondary_up'], self.content['croi']['secondary_down']]
    #
    #        if self.content['topo'] == 'wide':
    #            cl = [self.content['croi']['secondary']]
    #
    #        for cr in cl:
    #            cx, cy, r = cr
    #
    #            # (u0c, t0c) of trajectory passing through croi center
    #            u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
    #            t0c =   cx * np.cos(params['alpha']) + cy * np.sin(params['alpha'])
    #
    #            # local uc centered on croi
    #            uc = params['u0'] - u0c
    #
    #            # check whether trajectory enters cusps regions
    #            ## REVOIR ** 9/10 : en fait, il faut une adpatation aussi pour caustic ** -1/2
    #            ## REVOIR : on met ici 2 fois le rayon d'un cusp
    #            r = r + 2. * self.rcdict['extcroi'] * self.params['rho'] ** 0.9
    #            self._ccroi.append(np.array([cx, cy, r])) # for detailed plot
    #            if np.abs(uc) < r:
    #                Dt = np.sqrt(r**2 - uc**2)
    #                L.append(Interval(t0c - Dt, t0c + Dt))

    #        rcaus = self.rcdict['extcroi'] * self.params['rho']
            rcaus = np.max([0.003, 3. * self.params['rho']])
            rcusp = rcaus ** 0.9

            #    near caustics
            for c in np.ravel(cc.zetac).tolist():
                cx, cy = np.real(c), np.imag(c)

                # (u0c, t0c) of trajectory passing through croi center
                u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
                t0c =   cx * np.cos(params['alpha']) + cy * np.sin(params['alpha'])
                
                # local uc centered on croi
                uc = params['u0'] - u0c
                
                # check whether trajectory enters caustics regions
                self._ccroi.append(np.array([cx, cy, rcaus])) # for detailed plot
                if np.abs(uc) < rcaus:
                    Dt = np.sqrt(rcaus**2 - uc**2)
                    L.append(Interval(t0c - Dt, t0c + Dt))
            
            #    near cusps
            for key in self.content['cusps']:
                points = [(np.real(p), np.imag(p)) for p in self.content['cusps'][key]]

                for cx, cy in points:
                    # (u0c, t0c) of trajectory passing through croi center
                    u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
                    t0c =   cx * np.cos(params['alpha']) + cy * np.sin(params['alpha'])

                    # local uc centered on croi
                    uc = params['u0'] - u0c

                    # check whether trajectory enters cusps regions
                    self._ccroi.append(np.array([cx, cy, rcusp])) # for detailed plot
                    if np.abs(uc) < rcusp:
                        Dt = np.sqrt(rcusp**2 - uc**2)
                        L.append(Interval(t0c - Dt, t0c + Dt))

            #     basic general sampling
            t = self.rcdict['tmin']
            tex = list()
            dtmin = 1.e-4
            dtmax = 1.e-3
            while t < self.rcdict['tmax']:
                u = np.sqrt(self.params['u0'] ** 2 + t ** 2)
    #            if u > 0.8:
    #                dt = 0.05 * u
    #            else:
    #                dt = np.max([0.007 * u, 1.e-4])
                dt = 0.007 * u
                if dt > dtmax:
                    dt = dtmax
                if dt < dtmin:
                    dt = dtmin
                t += dt
                tex.append(t)

            tex = np.array(tex)
            self._tall = tex

            #    generate samling
            dtcroi = np.max([self.params['rho'] / 20., 7.e-6])
            R = Union(*L)
            intcroi, intout, intall = list(), list(), list()

            if not isinstance(R, EmptySet):
                
                #    sampling inside crois
                if isinstance(R, Interval):
                    intcroi.append(np.arange(R.left, R.right, dtcroi, dtype=np.float_))
    #                dt = np.max([dtcroi, np.abs(R.right - R.left) / self.rcdict['Nmaxcroi']])
    #                intcroi.append(np.arange(R.left, R.right, dt, dtype=np.float_))

                if isinstance(R, Union):
                    listcroi = [int for int in R.args]
                    for cr in listcroi:
                        intcroi.append(np.arange(cr.left, cr.right, dtcroi, dtype=np.float_))
    #                    dt = np.max([dtcroi, np.abs(cr.right - cr.left) / self.rcdict['Nmaxcroi']])
    #                    intcroi.append(np.arange(cr.left, cr.right, dt, dtype=np.float_))

                #    sampling outside crois
                S = R.complement(Interval(self.rcdict['tmin'], self.rcdict['tmax']))

                if isinstance(S, Interval):
                    tno = tex[(tex <= S.right) & (tex >=S.left)]
                    xno = -self.params['u0'] * np.sin(self.params['alpha']) + np.cos(self.params['alpha']) * tno
                    yno =  self.params['u0'] * np.cos(self.params['alpha']) + np.sin(self.params['alpha']) * tno
                    muno = np.array([aquad(self.params['s'], self.params['q'], self.params['rho'], xno[i], yno[i]) for i in range(len(tno))])
                    t, _ = self._optimizemc(tno, np.log10(muno), linerr=self.rcdict['linerr'])
                    intall.append(tno)
                    intout.append(t)

                if isinstance(S, Union):
                    listout = [int for int in S.args]
                    for out in listout:
                        tno = tex[(tex <= out.right) & (tex >= out.left)]
                        xno = -self.params['u0'] * np.sin(self.params['alpha']) + np.cos(self.params['alpha']) * tno
                        yno =  self.params['u0'] * np.cos(self.params['alpha']) + np.sin(self.params['alpha']) * tno
                        muno = np.array([aquad(self.params['s'], self.params['q'], self.params['rho'], xno[i], yno[i]) for i in range(len(tno))])
                        t, _ = self._optimizemc(tno, np.log10(muno), linerr=self.rcdict['linerr'])
                        intall.append(tno)
                        intout.append(t)

                flatcroi = list(itertools.chain(*intcroi))
                flatout = list(itertools.chain(*intout))
                flatall = list(itertools.chain(*intall))
                self._tall = np.sort(np.concatenate((flatall, flatcroi))) # for plotting checks
                tex = np.sort(np.concatenate((flatout, flatcroi)))

            # create non-optimized magnification curve
            self._xtraj = -self.params['u0'] * np.sin(self.params['alpha']) + np.cos(self.params['alpha']) * tex
            self._ytraj =  self.params['u0'] * np.cos(self.params['alpha']) + np.sin(self.params['alpha']) * tex
            try:
                # compute exact magnification and remove NaN and mu < 1 points
                muex = np.array([vbbmagU(self.params['s'], self.params['q'], self.params['rho'], -self._xtraj[i], self._ytraj[i], self.rcdict['vbbacc']) for i in range(len(tex))])
#                arg = (muex >= 1.)
                arg = np.logical_not(np.isnan(muex)) & (muex >= 1.)
                muex = muex[arg]
                tex = tex[arg]

                # optimize magnification curve in log(mu)
                logmuex = np.log10(muex)
                self._t, logmu = self._optimizemc(tex, logmuex, linerr=self.rcdict['linerr'])
                self._mu = 10 ** logmu

    #            # test 1 : fait apres coup
    #            muex = np.array([aquad(self.params['s'], self.params['q'], self.params['rho'], self._xtraj[i], self._ytraj[i]) for i in range(len(tex))])
    #
    #            logmuex = np.log10(muex)
    #            t, _ = self._optimizemc(tex, logmuex, linerr=self.rcdict['linerr'])
    #
    #            x = -self.params['u0'] * np.sin(self.params['alpha']) + np.cos(self.params['alpha']) * t
    #            y =  self.params['u0'] * np.cos(self.params['alpha']) + np.sin(self.params['alpha']) * t
    #            mu = np.array([vbbmagU(self.params['s'], self.params['q'], self.params['rho'], -x[i], y[i], self.rcdict['vbbacc']) for i in range(len(t))])
    #
    #            logmu = np.log10(mu)
    #            self._t, logmun = self._optimizemc(t, logmu, linerr=self.rcdict['linerr'])
    #            self._mu = 10 ** logmun

            except:
                # if fails, return A=1.
                self._t, self._mu = np.linspace(self.rcdict['tmin'], self.rcdict['tmax'], 4), np.ones(4, dtype=np.float)

            # create and store interpolation object
            mc = interp1d(self._t, self._mu, fill_value=1., bounds_error=False)
            self.content['mc'] = mc

            # store maximum magnification
            self.content['mumax'] = np.max(self._mu)
        
            # final summary
            printi(tcol + "Optimized magnification curve :" + tit + " {0:d} points, {1:.2e} < t/tE < {2:.3e}".format(len(self._t), np.min(self._t), np.max(self._t)) + tend)

    def _optimizemc(self, t, mu, linerr=0.001):
        """Optimize the sampling of the input curve
            
            This function removes points from the curve for which
            the difference from a linear approximation obtained from
            other points in the curve is not greater than a given threshold.
            
            Parameters
            ----------
            t : float array
                Orginial sampling in t.
            mu : float array
                Orginal mu(t) curve.
            linerr : float, otpional
                Threshold error in linear approximation mu(t_j) where
                t_j is the new sampling. Default value is: linerr=0.001
                
            Returns
            -------
            topt : float array
                Optimized sampling in t.
            muopt : float array
                Optimized mu(t) curve.
            """
        # set I/O shell display
        tend, tcol, tit = "\033[0m", "\033[0m\033[33m", "\033[0m\033[3m"
        
        # size of interval
        N = len(t)
        
        if N > 7:
            printi(tcol + "Optimizing magnification curve" + tit + " ({0:d} points, {1:.2e} < t/tE < {2:.3e})".format(N, np.min(t), np.max(t)) + tend)
            # algorithm
            ts = np.zeros(N, dtype=np.float_)
            As = np.zeros(N, dtype=np.float_)
            n = 0
            i = 0
            while n < N:
                cond = True
                As[n] = mu[i]
                ts[n] = t[i]
                n += 1
                p = 2
                while p <= N-1-i: # 2≤p
                    if np.logical_not(cond):
                        break
                    for k in np.arange(p-1)+1: # 1≤k≤p-1
                        Alin = (t[i+k] - t[i]) * (mu[i+p] - mu[i]) / (t[i+p] - t[i]) + mu[i]
                        cond = np.abs(Alin - mu[i+k]) <= linerr
                        if np.logical_not(cond):
                            i = i+p-1
                            break
                    p += 1
                if (p == N-i): break
            ts[n-1] = t[i]
            As[n-1] = mu[i]
            ts[n] = t[N-1]
            As[n] = mu[N-1]
            topt = copy(ts[0:n+1])
            muopt = copy(As[0:n+1])
        
            # verbose
            printi(tit + "  (" + str(n+1) + " points selected out of " + str(N) + ", gain: {:.1f}".format(100. - 100. * (n + 1) / N) + "% with accuracy " + str(linerr) + ")" + tend)
            
            return topt, muopt
                
        else:
            printi(tcol + "Optimizing magnification curve" + tend)
            printi(tit + "  (skipping)" + tend)
            
            return t, mu
        
#        printi(tcol + "Optimize magnification curve" + tit + " ({0:.2e} < t/tE < {1:.3e}) \n  (".format(np.min(t), np.max(t)) + str(n+1) + " points selected out of " + str(N) + ", gain: {:.1f}".format(100. - 100. * (n + 1) / N) + "% with accuracy " + str(linerr) + ")" + tend)


    def write(self, hdf5file, key, attrs=None, overwrite=False):
        """Write magnification curve (mc) to HDF5 file
            
            Parameters
            ----------
            hdf5file : string
                Output HDF5 file.
            key : string
                Key to store mc data (enventually 'path/keyword').
            overwrite : boolean, optional
                If True, overwrite HDF5 if it already exists.
            """
        # set I/O shell display
        tend, tcol, tit = "\033[0m", "\033[0m\033[34m", "\033[0m\033[3m"
        
        # verbose
        printi(tcol + "Write magnification curve " + tit + "'" + key + "'" + tcol + " in file " + tit + hdf5file + tend)
        
        # write mc
        aw = 'a'
        if overwrite: aw = 'w'
        with h5py.File(hdf5file, aw) as fmclib:
            themc = fmclib.create_dataset(str(key), data=[self._t, self._mu])
            # write parameters
            themc.attrs['s'] = self.params['s']
            themc.attrs['q'] = self.params['q']
            themc.attrs['u0'] = self.params['u0']
            themc.attrs['alpha'] = self.params['alpha']
            themc.attrs['rho'] = self.params['rho']
               
            # write information content
            themc.attrs['topo'] = self.content['topo']
            themc.attrs['mumax'] = self.content['mumax']
            for c in self.content['croi']:
                croiname = 'croi_' + c
                themc.attrs[croiname] = self.content['croi'][c]
               
            # write optional attributes
            if attrs:
                for attr in attrs:
                    sp = attr.split('/')
                    if len(sp) == 1:
                        fmclib.attrs[attr] = attrs[attr]
                    else:
                        nattr = sp.pop(len(sp) - 1)
                        apath = '/'.join(sp)
                        fmclib[apath].attrs[nattr] = attrs[attr]
            fmclib.flush()
            fmclib.close()

    def read(self, hdf5file, key):
        """Read magnification curve (mc) from HDF5 file
            
            Parameters
            ----------
            hdf5file : string
                Input HDF5 file.
            key : string
                Key to access mc data (enventually 'path/keyword').
            """
        # set I/O shell display
        tend, tcol, tit = "\033[0m", "\033[0m\033[32m", "\033[0m\033[3m"
        
        # read parameters
        with h5py.File(hdf5file, 'r') as fmclib:
            themc = fmclib[str(key)]
            self.params['s'] = themc.attrs['s']
            self.params['q'] = themc.attrs['q']
            self.params['u0'] = themc.attrs['u0']
            self.params['alpha'] = themc.attrs['alpha']
            self.params['rho'] = themc.attrs['rho']
            
            # read information content
            self.content['topo'] = themc.attrs['topo']
            self.content['mumax'] = themc.attrs['mumax']
            self.content['croi'] = dict()
            for c in ['central', 'secondary', 'secondary_up', 'secondary_down', 'resonant']:
                croiname = 'croi_' + c
                if croiname in themc.attrs:
                    self.content['croi'][c] = themc.attrs[croiname]
        
            # create interpolation object
            self._t = np.array(themc[0])
            self._mu = np.array(themc[1])
            self.content['mc'] = interp1d(self._t, self._mu, fill_value=1., bounds_error=False)
            fmclib.flush()
            fmclib.close()
    
        # verbose
        printi(tcol + "Read magnification curve " + tit + "'" + key + "'" + tcol + " in file " + tit + hdf5file + tend)
        printi(tit + "  (s={0:.4f}, q={1:.4e}, u0={2:.4e}, alpha={3:.4f}, rho={4:.4e})".format(self.params['s'], self.params['q'], self.params['u0'], self.params['alpha'], self.params['rho']) + tend)
    
    def plot(self, axis=None, figname=None, dots=True, croi=True):
        """Plot caustics, cusps positions and trajectory"""
        
        # initialize cc object
        cc = Caustics(self.params['s'], self.params['q'], N=256, cusp=True)
#        print cc.cusps

        # initialize plot
        plt.close('all')
        fig, (CAU, LC) = plt.subplots(2, sharex=True, figsize=(9,6))
        plt.tight_layout(pad=3.)
        plt.subplots_adjust(hspace=0.)
        fs = 12
        # plot magnification curve
#        LC.set_title('Magnification curve')
        LC.set_xlabel(r'$t$, $\xi$')
        LC.set_ylabel(r'$A$')
        if dots:
            LC.semilogy(self._t, self._mu, 'o-', linewidth=1)
        else:
            LC.semilogy(self._t, self._mu, '-', linewidth=1)
        LC.tick_params(labelsize=fs, width=0.8, direction='in', length=8)
        # plot caustics, trajectory + sampling, croi
        CAU.set_aspect('equal')
        CAU.tick_params(labelsize=fs, width=0.8, direction='in', length=8)
        CAU.set_title(r'Caustics $-$ caustics regions of interest $-$ magnification curve')
        CAU.set_xlim([-1.5, 1.5])
        CAU.set_ylim([-1.5, 1.5])
#        CAU.set_xlabel(r'$\xi$')
        CAU.set_ylabel(r'$\eta$')
        if axis:
            axis = np.array(axis)
            CAU.set_xlim([axis[0], axis[1]])
            CAU.set_ylim([axis[2], axis[3]])
        else:
            CAU.set_xlim([-1., 1.])
            CAU.set_ylim([-1., 1.])
#        for croi in self.content['croi']:
#            cx, cy, r = self.content['croi'][croi]
##            extr = r * self.rcdict['extcroi'] ## multiply radius
#            extr = r + self.rcdict['extcroi'] * params['rho'] ## add radius

        # plot caustic influence circles, caustics, lens positions
        if hasattr(self, '_ccroi'):
            for cx, cy, r in self._ccroi:
                ncx = cx * np.cos(-self.params['alpha']) - cy * np.sin(-self.params['alpha'])
                ncy = cx * np.sin(-self.params['alpha']) + cy * np.cos(-self.params['alpha'])
                CAU.scatter(ncx, ncy, marker='o', c='tomato', s=2)
                if croi:
                    trajcic = Circle((ncx, ncy), r, color='tomato', lw=0.5, ls='--', fill=False)
                    CAU.add_patch(trajcic)
        else:
            xc = np.real(cc.zetac) * np.cos(-self.params['alpha']) - np.imag(cc.zetac) * np.sin(-self.params['alpha'])
            yc = np.real(cc.zetac) * np.sin(-self.params['alpha']) + np.imag(cc.zetac) * np.cos(-self.params['alpha'])
            CAU.scatter(xc, yc, marker='o', c='tomato', s=2)
        
        AC2CM = self.params['s'] * self.params['q'] / (1. + self.params['q']) # TMP
        L1, L2 = -self.params['s'] + AC2CM, AC2CM
        CAU.scatter([L1, L2], [0., 0.], marker='+', c='midnightblue', s=80)
        
#        for i in range(len(self.cusps)):
#            c = np.mean(self.cusps[i])
#            rc = np.max(np.abs(self.cusps[i] - c))
#            if i == 0: ciccol = 'tomato'
#            else: ciccol = 'dimgrey'
#            # plot center of cic
#            CAU.scatter(np.real(c), np.imag(c), marker='o', c=ciccol, s=20)
#            # plot trajectory cic
#            trajcic = Circle((np.real(c), np.imag(c)), self.sigrc * rc, color=ciccol, lw=1, ls='--', fill=False)
#            CAU.add_patch(trajcic)
#        # plot caustics and cusps : DEJA DANS LES CROI
#        xc = np.real(cc.zetac) * np.cos(-self.params['alpha']) - np.imag(cc.zetac) * np.sin(-self.params['alpha'])
#        yc = np.real(cc.zetac) * np.sin(-self.params['alpha']) + np.imag(cc.zetac) * np.cos(-self.params['alpha'])
#        CAU.scatter(xc, yc, marker='.', c='red', s=0.5)
#        CAU.scatter(np.real(cc.zetac), np.imag(cc.zetac), marker='.', c='red', s=0.5)
#        [CAU.scatter(np.real(zetacu), np.imag(zetacu), marker='*', c='tomato', s=80) for zetacu in self.cusps]
        # plot lens positions, ref. CM
        # plot trajectory+source or sampling
#        trajcol = 'orange'
#        try:
#        if ptype == 'sampling':

        # plot mc non-optimized trajectory
        if hasattr(self, '_tall'):
            CAU.scatter(self._tall, np.ones_like(self._tall) * self.params['u0'], marker='o', c='orange', s=1)
#        xtraj = self._xtraj * np.cos(-self.params['alpha']) - self._ytraj * np.sin(-self.params['alpha'])
#        ytraj = self._xtraj * np.sin(-self.params['alpha']) + self._ytraj * np.cos(-self.params['alpha'])
#        CAU.scatter(xtraj, ytraj, marker='o', c=trajcol, s=1)
#            CAU.scatter(self._xtraj, self._ytraj, marker='.', c=trajcol)

        # plot mc optimized trajectory
        CAU.scatter(self._t, np.ones_like(self._t) * self.params['u0'], marker='o', c='brown', s=4)
#            xtraj = -self.params['u0'] * np.sin(self.params['alpha']) + np.cos(self.params['alpha']) * self._t
#            ytraj =  self.params['u0'] * np.cos(self.params['alpha']) + np.sin(self.params['alpha']) * self._t
#            CAU.scatter(xtraj, ytraj, marker='.', c='brown', s=8)

#        except:
#            pass
#        if ptype == 'trajectory':
#            # plot trajectory
#            ttraj = np.linspace(-4., 4., 50)
#            xtraj = -self.params['u0'] * np.sin(self.params['alpha']) + np.cos(self.params['alpha']) * ttraj
#            ytraj =  self.params['u0'] * np.cos(self.params['alpha']) + np.sin(self.params['alpha']) * ttraj
#            CAU.arrow(xtraj[0], ytraj[0], xtraj[len(xtraj) - 1] - xtraj[0], ytraj[len(ytraj) - 1] - ytraj[0], shape='full', lw=0.8, head_width=.1, color=trajcol, edgecolor=trajcol, facecolor=trajcol, fill=True)
#            # plot source positions
#            listcs = xtraj + 1j * ytraj
#            listspos = [Circle((np.real(cs), np.imag(cs)), self.params['rho'], facecolor='gold', edgecolor=None, alpha=1., fill=True) for cs in listcs]
#            [CAU.add_patch(spos) for spos in listspos]
        if figname:
            plt.savefig(figname)
        else:
            plt.show()


class LightCurve():
    """Light curve objets
        (child class of MagnificationCurve)
        :call:
        lc = LightCurve(data)
        :inputs:
        mc, a MagnificationCurve object.
        data, data files.
        :returns:
        lc.chi2, best-fit chi2.
        :user methods:
        lc.fit(), fit data to mc.
        lc.plot(), plot light cuve and data.
        
        dmag = estimation of min(delta(mag)) of the data (default: 0.)
        """
    def __init__(self, datasets, trange=None, dmag=None):
        
        # set I/O shell display
        tun, tcol, tend, tit = "\033[0m\033[1;33m", "\033[0m\033[33m", "\033[0m", "\033[0m\033[3m"
        
        # lc parameters and information
        self.params = dict()
        self.content = dict()
        
        # read data sets
        self._datalist = list()
        self.content['dmag'] = 0.
        for datai in datasets:
            # check if input file exists
            if not os.path.isfile(datai):
                raise IOError("data file '" + datai + "' is missing")
            
            # read data files
            df = pd.read_csv(datai, sep=' ', header=None, names=['hjd', 'mag', 'errmag'], usecols=[1, 2, 3])
            hjd = df['hjd'].values
            if hjd[0] > 2450000.:
                hjd = hjd - 2450000.
            mag = df['mag'].values
            errmag = df['errmag'].values
            flux = np.power(10., - mag / 2.5)
            errflux = flux * np.log(10.) / 2.5 * errmag
            if not trange:
                self._datalist.append(tuple([len(hjd), hjd, mag, errmag, flux, errflux, datai]))
            else:
                arg = np.ravel(np.argwhere((trange[0] < hjd) & (hjd < trange[1])))
                self._datalist.append(tuple([len(hjd[arg]), hjd[arg], mag[arg], errmag[arg], flux[arg], errflux[arg], datai]))
            
            if dmag:
                # estimate empirical ∆mag
                serrmag = np.sort(errmag)
                maxerr = np.percentile(serrmag, 30)
                arg = np.argwhere(errmag < maxerr)
                smag = mag[arg]
                omag = np.sort(smag)
                mmin, mmax = np.percentile(omag, [1, 99])
                printd(tcol + "Estimate baseline, current peak and delta(mag) for " + tit + datai + "\n  (" + str(mmin) + ", " + str(mmax) + ", " + str(np.abs(mmax - mmin)) + ")" + tend)
                if self.content['dmag'] < np.abs(mmax - mmin):
                    self.content['dmag'] = np.abs(mmax - mmin)
        printi(tcol + "Discarding magnification curves with " + tit + "delta(mag) < " + str(self.content['dmag']) + tend)
        
        # run command keywords
        self.rcdict = {
            'tol': 0.07,            # tolerance on chi2/dof to stop minimization
            'maxiter': 70,          # maximum number of iterations in minimization
            }

    def rc(self, key, *value):
        """Set run commands"""
        # set I/O shell display
        tbf, tcol, tend, tit = "\033[1m", "\033[31m", "\033[0m", "\033[3m"
        
        # check whether run command exists
        if key not in self.rcdict:
            raise KeyError("run command keyword '" + key + "' does not exist")
        
        # if argument value exists, set new value, otherwise display value
        if value:
            self.rcdict[key] = value[0]
        printi(tcol + "Run command (light curve) " + tend + tit + key + tend + tcol + " is set to " + tend + tit + str(self.rcdict[key]) + tend)
    
    def fit(self, mc, croi, ori, init=None):
        """Find best-fit parameters (tE, t0, FS, Fb)
            
            1610 :
            7950., 50.
            
            0060 :
            7147.5, 73.7
            
            1737 :
            7265., 30.
            
            init=[(ta,tE), ...]
            
            *)--> faire un croi automatique lu dans mc, finalement
            mettre refcroi dans la mc directment
            
            *) mettre Nmc dans la gengridlibs car sinon info disparait de la lib
            """
        # set I/O shell display
        tbf, tcol, tend, tit = "\033[0m\033[1;35m", "\033[0m\033[35m", "\033[0m", "\033[0m\033[3m"
        
        # check arguments
        if ori not in ['+', '-']:
            raise ValueError("unknown value for argument 'ori' (choose '+' or '-')")
        if croi not in ['central', 'secondary']:
            raise ValueError("unknown value for argument 'croi' (choose 'central' or 'secondary')")
        
        # current existing parameters are taken from mc
        self.params = mc.params
        
        # load magnification curve interp1d function (F(t) = mcf(t) * Fs + Fb)
        self.content['mc'], self._t, self._mu = mc.content['mc'], mc._t, mc._mu

        # verbose
        printi(tcol + "Fitting model to " + tit + str(len(self._datalist)) + tcol + " dataset(s)" + tend)

        # fit or not fit (tE, t0)
        if not init:
            t0f = self.params['t0']
            tEf = self.params['tE']
        else:
            # get reference croi
            if mc.content['topo'] == 'interm':
                cx, cy, _ = mc.content['croi']['resonant']
            if mc.content['topo'] == 'close':
                if croi == 'secondary':
                    cx, cy, _ = mc.content['croi']['secondary_up']
                else:
                    cx, cy, _ = mc.content['croi']['central']
            if mc.content['topo'] == 'wide':
                cx, cy, _ = mc.content['croi'][croi]
            # t0 for trajectory passing by reference croi center in Einstein units
            tc = cx * np.cos(self.params['alpha']) + cy * np.sin(self.params['alpha'])

            # prepare fit parameters
            if type(init) == tuple:
                init = [init]
            chit = np.inf
            for tatE in init:
                ta, tE = tatE
    #            print tbf + "test: ta, tE", ta, tE, tend
                if ori == '+': # trajectory towards right
                    tEi = tE
                else: # trajectory towards left
                    tEi = - tE
                # observed anomalous date ta converions to t0
                t0i = ta - tc * tEi
                # fit  to find tE and t0
                fit = opt.minimize(self._ffit, np.array([tEi , t0i]), method='nelder-mead', tol=self.rcdict['tol'], options={'maxiter': self.rcdict['maxiter']})
                # check if fit is better
    #            print "chi2/dof", fit.fun
                if fit.fun < chit:
    #                print "chosen", fit.fun
                    chit = fit.fun
                    tEf = fit.x[0]
                    t0f = fit.x[1]
    #        print tEf, t0f

        # compute (Fs, Fb), save best fit parameters and chi2
        self.params['Fs'], self.params['Fb'], self.content['datasets'] = list(), list(), list()
        globchi2, globN = 0., 0
        for dat in self._datalist:
            mu = self.content['mc']((dat[1] - t0f) / tEf)

            Fs, Fb, chi2 = self._wreglin(mu, dat[4], dat[5])

#            Fs, Fb, rv, pv, err = stats.linregress(mu, dat[4])
#            chi2 = np.sum((dat[4] - (mu * Fs + Fb))**2 / dat[5]**2)

            # save chi2 and datasets properties (name, number of data, fit chi2)
            if Fs > 0.:
                self.content['datasets'].append((dat[6], dat[0], chi2))
            else:
                self.content['datasets'].append((dat[6], dat[0], np.inf))
# tests : plot and print
#            plt.plot(mu, dat[4], '+')
#            plt.plot(mu, mu * Fs + Fb)
#            plt.show()
            # save (Fs, Fb)
            self.params['Fs'].append(Fs)
            self.params['Fb'].append(Fb)
            # save t0
            self.params['t0'] = t0f
# tests: (Fs, Fb)
#        print self.params['Fs'], self.params['Fb']
            # global chi2 and number of data
            globchi2 += chi2
            globN += dat[0]

        # save global chi2 and number of data
        self.content['chi2'] = [globchi2, globN]
        # update params only if (t0, tE) were fitted
        if tEf < 0.:
            # if tE < 0, update parameters to match π < alpha < 3π/2
            self.params['tE'] = - tEf
            self.params['alpha'] = np.pi + self.params['alpha']
            self.params['u0'] = - self.params['u0']
            self._t = - self._t
        else:
            self.params['tE'] = tEf

        # fit summary
        printi(tit + "  (t0={0:4.5f}, tE={1:.2f}, s={2:.4f}, q={3:.4e}, u0={4:.4e}, alpha={5:.4f}, rho={6:.4e}, chi2={7:.3e}, chi2/dof={8:.3e}".format(self.params['t0'], self.params['tE'], self.params['s'], self.params['q'], self.params['u0'], self.params['alpha'], self.params['rho'], globchi2, globchi2/globN) + ")" + tend)
        for i in range(len(self._datalist)):
            printd(tit + "  (" + str(self._datalist[i][6]) + " with " + str(self._datalist[i][0]) + " data points)" + tend)

    def _ffit(self, x):
        """Returns model chi2/dof"""
        globN, globchi2 = 0, 0.
        for dat in self._datalist:
            mu = self.content['mc']((dat[1] - x[1]) / x[0])
            
            Fs, Fb, chi2 = self._wreglin(mu, dat[4], dat[5])

#            Fs, Fb, rv, pv, err = stats.linregress(mu, dat[4])
#            chi2 = np.sum((dat[4] - (mu * Fs + Fb))**2 / dat[5]**2)

            globchi2 += chi2
            globN += dat[0]
        return globchi2 / globN

#    def _wreglin(self, x, y, err):
#        """Weighted linear fit"""
#        w = 1. / err**2
#        W = np.sum(w)
#        Sx = np.average(x, weights=w)
#        Sy = np.average(y, weights=w)
#        Sxx = np.average(x**2, weights=w)
#        Sxy = np.average(x * y, weights=w)
#        Fs = (W * Sxy - Sx * Sy) / (W * Sxx - Sx**2)
#        Fb = Sy - Sx * Fs
##        siga = np.sqrt((1. + Sx**2/Sdxx) / W)
##        sigb = np.sqrt(1. / (W * Sdxx))
##        covab = - Sx / (W * Sdxx)
#        chi2 = np.sum(w * (y - (Fb + Fs * x))**2)
#        return Fs, Fb, chi2

    def _wreglin(self, x, y, err):
        ## a verifier sur un exemple
        """Weighted linear fit"""
        w = 1. / err**2
        W = np.sum(w)
        Sx = np.average(x, weights=w)
        Sy = np.average(y, weights=w)
        dx = x - Sx
        dy = y - Sy
        Sdxx = np.average(dx**2, weights=w)
        Sdxy = np.average(dx * dy, weights=w)
        Fs = Sdxy / Sdxx
        Fb = Sy - Sx * Fs
#        siga = np.sqrt((1. + Sx**2/Sdxx) / W)
#        sigb = np.sqrt(1. / (W * Sdxx))
#        covab = - Sx / (W * Sdxx)
        chi2 = np.sum(w * (y - (Fb + Fs * x))**2)
        return Fs, Fb, chi2

    def plot(self):
        """Plot light curve
            A REVOIR COPLETEMENT"""
#        # set correct tE
#        if np.pi / 2. < self.params['alpha'] % (2 * np.pi) < 3. * np.pi / 2.:
#            tEf = - self.params['tE']
#        else:
#            tEf = self.params['tE']
        plt.close('all')
        fig, LC = plt.subplots(1, figsize=(8,5))
        LC.invert_yaxis()
        # reference light curve is the first of the list
        flux = self._mu * self.params['Fs'][0] + self.params['Fb'][0]
        t = self._t * self.params['tE'] + self.params['t0']
        LC.plot(t, -2.5 * np.log10(flux), '-', color='darkred', linewidth=1)
        LC.plot(t, -2.5 * np.log10(flux), '+', color='darkred')
        # data
        i = 0
        for dat in self._datalist:
            # deblending
            magnif = (np.power(10., - dat[2] / 2.5) - self.params['Fb'][i]) / self.params['Fs'][i]
            flux = magnif * self.params['Fs'][0] + self.params['Fb'][0]
            LC.errorbar(dat[1], - 2.5 * np.log10(flux), yerr=dat[3], fmt='o', markersize=3, linewidth=1)
            i += 1
        plt.show()


def hdf5_show(hdf5file):
    """Show content of HDF5 file collection of magnification curves
    """
    fmclib = h5py.File(hdf5file, 'r')
    
    for gr in fmclib.keys():
        for n in fmclib[gr].keys():
            printi("a faire")
            # faire le bilan du nombre de keys, donner celles manquantes
            # donner pour chaque key la valeur de (s,q)
            # faire un tracé des cautiques et de toutes les trajectoires

    fmclib.close()
    
def aquad(s, q, rho, x, y):
    """Finite-source magnification up to the quadrupole term.
    """
    # no limb-darkening
    Gamma = 0.
    
    # source center
    zeta0 = np.complex(x, y)
    zeta0 = zeta0 - np.complex(s * q / (1. + q), 0.)
    z0 = _solvelenseq(s, q, zeta0)
    
    # compute quadrupole
    W1 = 1. / (1. + q) * (1. / z0 + q / (z0 + s))
    z0 = z0[np.abs(z0 - W1.conjugate() - zeta0) < 0.000001]
    nr = len(z0)
    Wk = np.empty((7, nr), dtype=np.complex128)
    Wk[2] = -1. / (1. + q) * (1. / z0**2 + q / (z0 + s)**2)
    Wk[3] = 2. / (1. + q) * (1. / z0**3 + q / (z0 + s)**3)
    Wk[4] = -6. / (1. + q) * (1. / z0**4 + q / (z0 + s)**4)
    A0, A2 = quadrupole(Wk, rho, Gamma)
    
    return A2


if __name__ == '__main__':
    verbosity('DEBUG')

    # visionner des mcgrid
    mc = MagnificationCurve()
    #    hdf5file = '../mcgrids/mcgrid7/mcgrid7_10.hdf5'
    hdf5file = '../corbeille/mcgrid25_20.hdf5'

    fmclib = h5py.File(hdf5file, 'r')
    
    if '0/0' not in fmclib:
        printi("not in")

#    for gr in fmclib.keys():
#        for n in fmclib[gr].keys():
#            mc.read(hdf5file, gr + '/' + n)
#            mc.plot(dots=False, croi=False)
    fmclib.close()

#### test des routines
#    mc = MagnificationCurve()
##    params = {'s': 0.5848, 'q': 1.0000e-05, 'u0': -1.0755, 'alpha': 1.1906, 'rho': 6.8660e-03}
##    params = {'s': 0.916494, 'q': 0.0333775, 'u0': 0.135853, 'alpha': 0.191671, 'rho': 0.0177671}
##    params = {'s': 0.916494, 'q': 0.0333775, 'u0': -0.135853, 'alpha': 2.94992165, 'rho': 0.0177671}
##    params = {'s': 1., 'q': 1., 'u0': 0., 'alpha': 1.55, 'rho': 0.007}
##    params = {'s': 1., 'q': 1., 'u0': 0.1625, 'alpha': 0.5, 'rho': 1.e-2}
##    params = {'s': 1., 'q': 1., 'u0': 0.1625, 'alpha': 0.5, 'rho': 1.e-3}
##    params = {'s': 1., 'q': 1., 'u0': 0.1625, 'alpha': 0.5, 'rho': 1.e-4}
##    params = {'s': 1., 'q': 1., 'u0': -0.104, 'alpha': 0.3, 'rho': 0.007}
##    params = {'s': 1.6, 'q': 0.01, 'u0': 0.02, 'alpha': 0.5, 'rho': 0.007}
#    params = {'s': 3.2, 'q': 0.1, 'u0': 0.1, 'alpha': 0., 'rho': 0.007}
##    params = {'s': 3.2, 'q': 0.1, 'u0': 0.02, 'alpha': 0., 'rho': 0.007}
##    params = {'s': 0.500000, 'q':  0.3, 'u0':  -4.1096e-03, 'alpha':  3.362138, 'rho': 6.1210e-04}
##    params = {'s': 0.200000, 'q':  1.0000e+00, 'u0':  -2.1096e-02, 'alpha':  3.362138, 'rho': 6.1210e-04}
##    params = {'s': 0.800000, 'q':  4.e-05, 'u0':  -1.e-05, 'alpha':  3.362138, 'rho': 1.1210e-04}
#
#    mc.create(params)
#    mc.plot(dots=False, croi=False)
