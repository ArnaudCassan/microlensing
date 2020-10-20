# -*- coding: utf-8 -*-
"""Binary lens caustics, cusps and critical curves (module: acmltools/caustics)"""

# Copyright (c) 2017 Arnaud Cassan
# Distributed under the terms of the MIT license
#
# This module [caustics.py]:
#       Binary lens caustics, critical curves and cusps
#
# This module is part of the microlensing modelling tools suite:
#       Gravitational microlensing tools by A.C. (acmltools)
#       https://github.com/ArnaudCassan/acmltools

# Part of this code is based on publication:
#   Cassan, A. (2018), Efficient grid search in microlensing: binary-lens
#       local solutions and initial model parameters guesses, in prep.

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import copy
import scipy.optimize as opt
import warnings
from mpmath import mp
from scipy.interpolate import interp1d

class Caustics():
    """Binary-lens caustics, critical curves and cusps

        Conventions:
        - Origin of the coordinate system at the center of mass
          of the two bodies.
        - Least massive body (i.e. secondary) located on the left
          hand side.

        Calling Caustics
        ================
        cc = Caustics(s, q, N=200, cusp=False)

        Parameters
        ----------
        s : float
            Binary lens separation (Einstein units).
        q : float
            Binary lens mass ratio (q ≤ 1).
        N : int, optional
            Number of points used to draw caustics and critical curves.
            Default is: N=200. To test topology (without computing
            critical curves and caustics), choose N=None. Cusps
            can be computed even with N=None.
        cusp : boolean, optional
            If True, computes the coordinates of the cusps.
            Default value is: False.

        Attributes
        ----------
        zetac : complex 1-D array, or None
            Caustics points, or None if N=None.
        zc : complex 1-D array, or None
            Critical curves points, or None if N=None.
        topo : str
            Binary lens topology. Values are: 'close', 'interm', 'wide'
        cusps : dictionary of complex arrays
            Contains the coordinates of the cusps if cusp=True.
            Keywords are:
            - 'central' (for close and wide binary),
            - 'secondary' (for wide binary),
            - 'secondary_up', 'secondary_down' (for close binary),
            - 'resonant' (for intermediate binary).
        Methods
        -------
        pltcrit : see below.
        pltcaus : see below.

        Examples
        --------
        >>> cc = Caustics(1.4, 0.1)
        >>> cc.pltcaus()
        >>> cc.pltcrit(axis=[-4, 4, -1, 2], figname='critcurves.pdf')

        >>> cc = Caustics(1.4, 0.1, N=None)
        >>> print cc.topo
        'interm'

        >>> cc = Caustics(0.3, 0.0001, N=400, cusp=True)
        >>> cc.pltcaus()
        >>> for key in cc.cusps:
                print key, cc.cusps[key]
        secondary_up [ 3.03254412-0.06355483j  3.03291176-0.06355996j
                3.03272425-0.06384869j]
        central [  1.22467386e-05 +0.00000000e+00j   3.07584367e-06
                -1.91770553e-05j -3.12122682e-05 +0.00000000e+00j
                3.07584367e-06 +1.91770553e-05j]
        secondary_down [ 3.03254412+0.06355483j  3.03291176+0.06355996j
                3.03272425+0.06384869j]
        """
    def __init__(self, s, q, N=200, cusp=False):
        self.s = s
        self.q = q
        self._N = N
        self.extraprec = 80 # common extra precision for mp.polyroots
        mp.dps = 40 # precision mpmath
        self._cusp = cusp
        self.cusps = dict()
        self.zc = None
        self.zetac = None
        self._zcu = None
        self._AC2CM = s * q / (1. + q)
        self._caulabels = [r'$\xi \:[\theta_{\rm E}]$', r'$\eta \:[\theta_{\rm E}]$']
        self._crilabels = [r'$x \:[\theta_{\rm E}]$', r'$y \:[\theta_{\rm E}]$']
        self.s_lower_lim = 0.1  # lower limit of separation s
        self.s_higher_lim = 10. # higher limit of separation s
        self.smin, self.smax = self._findtopo()
        if N: self._critcauscurves()
        if cusp: self._getcusps()

    def __repr__(self):
        return "Caustics object with parameters s = " + str(self.s) + " and q = " + str(self.q)

    def __str__(self):
        return "Caustics object with parameters s = " + str(self.s) + " and q = " + str(self.q)

    def _critcauscurves(self):
        """Compute critical curves and caustics"""
        # critical curves, ref. AC
        zc = list()
        for phi in np.linspace(0., 2.*np.pi, self._N, endpoint=False):
            witt = [1., 2. * self.s, self.s**2 - np.exp(1j * phi),
                    -2. * self.s * np.exp(1j * phi) / (1. + self.q), -(self.s**2 * np.exp(1j * phi) / (1. + self.q))]
            zc.append(np.roots(witt))
        zc = np.array(zc)
        # critical curves, ref. CM
        self.zc = zc + self._AC2CM
        # caustics, ref. CM
        self.zetac = zc - (1. / np.conj(zc) + self.q / (np.conj(zc) + self.s)) / (1. + self.q) + self._AC2CM

    def _findtopo(self):
        """Find binary lens topology"""
        # limits between topologies
        a3 = 1.
        a2 = 3. / (1. + self.q)
        a1 = 3. / (1. + self.q)
        a0 = 1. / (1. + self.q)
        zn = np.roots([a3, a2, a1 , a0])
        W2 = -1. / (1. + self.q) * (1. / zn**2 + self.q / (zn + 1.)**2)
        sn = np.sqrt(np.abs(W2))
        c2i = np.min(sn)
        i2w = np.max(sn)
        # find topology and the separation s range value of the topology
        if self.s < c2i:
            self.topo = 'close'
            sn_chgt_min = self.s_lower_lim
            sn_chgt_max = np.min(sn)
        if c2i <= self.s <= i2w:
            self.topo = 'interm'
            sn_chgt_min = np.min(sn)
            sn_chgt_max = np.max(sn)
        if i2w < self.s:
            self.topo = 'wide'
            sn_chgt_min = np.max(sn)
            sn_chgt_max = self.s_higher_lim
        return (sn_chgt_min, sn_chgt_max)

    def _getcusps(self):
        """Get cusps coordinates"""
        # precision parameters
        maxsteps = 300   # max iterations of mp.findroot
        tol = 1e-20    # tolerance in mp.findroot
        # compute (Ox) roots
        witt = [1., 2. * self.s, self.s**2 - 1., -2. * self.s / (1. + self.q), -(self.s**2 / (1. + self.q))]
        zc = np.roots(witt)
        # topology: close
        if self.topo == 'close':
            self._yl = []
            method = 'illinois'
            dico_interp = self._enveloppes()
            # (Ox)
            arg = np.argsort(np.imag(zc))
            arg = arg[[1, 2]]
            zc = zc[arg]
            arg = np.argsort(np.real(zc))
            sort = zc[arg]
            ## central caustic
            # A, b3, [1, 1+q] -> Witt
            ccA = complex(np.real(sort[1]))
            # C, b4, [1, 1+1/q] -> Witt
            ccC = complex(np.real(sort[0]))
            # B, b3, [0, 1] -> poly
            loglbdmin, loglbdmax = dico_interp['cb3B']  # Use interpolation on the envelopes to find the optimize search interval of lambda
            loglbdmax = np.log10(1.)
            self._br = 2
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)   # Use mpmath to compute lbd near 1
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccB = complex(bn[2]) * self.s
            self._yl.append((3, lbd, ccB))
            # D -> conj(B)
            ccD = np.conj(ccB)
            ## secondary caustics
            bn, sn = self._sortbnccu(0.)
            zsadd = complex(bn[0])
            # E, b1, [0, ∞] -> poly
            loglbdmin, loglbdmax = dico_interp['cb1']   # Use interpolation on the envelopes to find the optimize search interval of lambda
            self._br = 0
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccE = complex(bn[0])
            self._yl.append((1, lbd, ccE))
            ccE = ccE * self.s
            # H -> conj(E)
            ccH = np.conj(ccE)
            # F, b3, [1+q, ∞] -> poly
            loglbdmin, loglbdmax = dico_interp['cb3F']    # Use interpolation on the envelopes to find the optimize search interval of lambda
            self._br = 2
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccF = complex(bn[2])
            self._yl.append((3, lbd, ccF))
            ccF = ccF * self.s
            # I -> conj(F)
            ccI = np.conj(ccF)
            # G, b5, [1+1/q, ∞] -> poly
            loglbdmin, loglbdmax = dico_interp['cb5']    # Use interpolation on the envelopes to find the optimize search interval of lambda
            self._br = 4
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccG = complex(bn[4])
            self._yl.append((5, lbd, ccG))
            ccG = ccG * self.s
            # J -> conj(G)
            ccJ = np.conj(ccG)
            ## store cusps
            zcu = np.array([ccA, ccB, ccC, ccD])
            self.cusps['central'] = zcu - 1. / (1. + self.q) * (1. / np.conj(zcu) + self.q / (np.conj(zcu) + self.s)) + self._AC2CM
            zcu = np.array([ccE, ccF, ccG])
            self.cusps['secondary_up'] = zcu - 1. / (1. + self.q) * (1. / np.conj(zcu) + self.q / (np.conj(zcu) + self.s)) + self._AC2CM
            zcu = np.array([ccH, ccI, ccJ])
            self.cusps['secondary_down'] = zcu - 1. / (1. + self.q) * (1. / np.conj(zcu) + self.q / (np.conj(zcu) + self.s)) + self._AC2CM
            # for cusp curves only
            self._zcu = np.array([ccA, ccB , ccC, ccD, ccE, ccF, ccG, ccH, ccI, ccJ]) + self._AC2CM
        # topology: intermediate
        if self.topo == 'interm':
            self._yl = []
            method = 'illinois'
            dico_interp = self._enveloppes()
            # (Ox)
#            zc = copy(self.zc[0] - self._AC2CM)
            arg = np.argsort(np.imag(zc))
            arg = arg[[1, 2]]
            zc = zc[arg]
            arg = np.argsort(np.real(zc))
            sort = zc[arg]
            ## resonant caustic
            # A, b3, [1, 1+q] -> Witt
            ccA = complex(np.real(sort[1]))
            # D, b4, [1, 1+1/q]  -> Witt
            ccD = complex(np.real(sort[0]))
            # B, b3, [1+q, ∞]
            loglbdmin, loglbdmax = dico_interp['cb3']    # Use interpolation on the envelopes to find the optimize search interval of lambda
            loglbdmin = np.log10(1. + self.q)
            self._br = 2
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccB = complex(bn[2]) * self.s
            self._yl.append((3, lbd, ccB))
            # F -> conj(B)
            ccF = np.conj(ccB)
            # C, b5, [1+1/q, ∞]
            loglbdmin, loglbdmax = dico_interp['cb5']    # Use interpolation on the envelopes to find the optimize search interval of lambda
            loglbdmin = np.log10(1. + 1./self.q)
            self._br = 4
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccC = complex(bn[4])
            self._yl.append((5, lbd, ccC))
            ccC = ccC * self.s
            # E -> conj(C)
            ccE = np.conj(ccC)
            ## store cusps
            zcu = np.array([ccA, ccB, ccC, ccD, ccE, ccF])
            self.cusps['resonant'] = zcu - 1. / (1. + self.q) * (1. / np.conj(zcu) + self.q / (np.conj(zcu) + self.s)) + self._AC2CM
            # for cusp curves only
            self._zcu = zcu + self._AC2CM
        # topology: wide
        if self.topo == 'wide':
            self._yl = []
            method = 'illinois'
            dico_interp = self._enveloppes()
            # (Ox)
#            zc = copy(self.zc[0] - self._AC2CM)
            arg = np.argsort(np.real(zc))
            sort = zc[arg]
            ## central caustic
            # A, b3, [1, 1+q] -> Witt
            ccA = complex(np.real(sort[3]))
            # C, b6, [0, 1+q] -> Witt
            ccC = complex(np.real(sort[2]))
            # B, b3, [1+q, ∞] -> poly
            loglbdmin, loglbdmax = dico_interp['cb3']    # Use interpolation on the envelopes to find the optimize search interval of lambda
            loglbdmin = np.log10(1. + self.q)
            self._br = 2
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccB = complex(bn[2]) * self.s
            self._yl.append((3, lbd, ccB))
            # D -> conj(B)
            ccD = np.conj(ccB)
            ## secondary caustic
            # E, b5,  [0, 1+1/q] -> Witt
            ccE = complex(np.real(sort[1]))
            # G, b4, [1, 1+1/q] -> Witt
            ccG = complex(np.real(sort[0]))
            # F, b5, [1+1/q, ∞] -> poly
            loglbdmin, loglbdmax = dico_interp['cb5']    # Use interpolation on the envelopes to find the optimize search interval of lambda
            loglbdmin = np.log10(1. + 1./self.q)
            self._br = 4
            loglbd = mp.findroot(self._fpoly, (mp.mpf(loglbdmin), mp.mpf(loglbdmax)), tol=tol, maxsteps=maxsteps, solver=method)
            lbd = mp.power(10, loglbd)
            bn, sn = self._sortbnccu(lbd)
            ccF = complex(bn[4])
            self._yl.append((5, lbd, ccF))
            ccF = ccF * self.s
            # H -> conj(F)
            ccH = np.conj(ccF)
            ## store cusps
            zcu = np.array([ccA, ccB, ccC, ccD])
            self.cusps['central'] = zcu - 1. / (1. + self.q) * (1. / np.conj(zcu) + self.q / (np.conj(zcu) + self.s)) + self._AC2CM
            zcu = np.array([ccE, ccF, ccG, ccH])
            self.cusps['secondary'] = zcu - 1. / (1. + self.q) * (1. / np.conj(zcu) + self.q / (np.conj(zcu) + self.s)) + self._AC2CM
            # for cusp curves only
            self._zcu = np.array([ccA, ccB , ccC, ccD, ccE, ccF, ccG, ccH]) + self._AC2CM

    def _fpoly(self, loglbd):
        """Sub-routine of _getcusps"""
        lbd = mp.power(10, loglbd)
        bn, sn = self._sortbnccu(lbd)
        return sn[self._br] - self.s

    def _enveloppes(self):
        interp = dict()
        if self.topo == 'close':
            # Envelopes data for the 4 cusps compute with brentq
            env_logqb3B = np.log10([1.00000000e-06, 1.32571137e-06, 1.75751062e-06, 2.32995181e-06, 3.08884360e-06, 4.09491506e-06, 5.42867544e-06, 7.19685673e-06, 9.54095476e-06, 1.26485522e-05, 1.67683294e-05, 2.94705170e-05, 3.90693994e-05, 5.17947468e-05, 6.86648845e-05, 9.10298178e-05, 1.20679264e-04, 1.59985872e-04, 2.12095089e-04, 2.81176870e-04, 3.72759372e-04, 4.94171336e-04, 6.55128557e-04, 8.68511374e-04, 1.15139540e-03, 1.52641797e-03, 2.02358965e-03, 2.68269580e-03, 3.55648031e-03, 4.71486636e-03, 6.25055193e-03, 8.28642773e-03, 1.09854114e-02, 1.45634848e-02, 1.93069773e-02, 2.55954792e-02, 3.39322177e-02, 4.49843267e-02, 5.96362332e-02, 7.90604321e-02, 1.04811313e-01, 1.38949549e-01, 1.84206997e-01, 2.44205309e-01, 3.23745754e-01, 4.29193426e-01, 5.68986603e-01, 7.54312006e-01, 1.00000000e+00])   #addition
            env_logq = np.log10([1.00000000e-06, 3.08884360e-06, 9.54095476e-06, 3.90693994e-05, 1.20679264e-04, 3.72759372e-04, 1.15139540e-03, 3.55648031e-03, 1.09854114e-02, 3.39322177e-02, 1.04811313e-01, 3.23745754e-01, 1.00000000e+00])    #addition
            env_loglbd_haute_cb3B = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])   #addition
            env_loglbd_basse_cb3B = np.array([ -8.92194201,  -8.96286492,  -9.00378859,  -9.04473246, -9.08568671,  -9.12665982,  -9.16764463,  -9.2086472 , -9.24965448,  -9.29068069,  -9.33175046,  -9.41392665, -9.45506969,  -9.49622743,  -9.53741902,  -9.57864262, -9.61991217,  -9.66122095,  -9.70257049,  -9.74396351, -9.78539816,  -9.82689222,  -9.86842857,  -9.91003274, -9.95168718,  -9.99339133, -10.03514373, -10.07693977, -10.11876938, -10.16061995, -10.20246093, -10.24427005, -10.28600572, -10.3275982 , -10.36897138, -10.41002659, -10.4506081 , -10.49055218, -10.5296027 , -10.56746895, -10.60376846, -10.63804942, -10.66975838, -10.69829209, -10.72295903, -10.74308385, -10.75800408, -10.76720213, -10.77031194])   #addition
            env_loglbd_haute_cb1 = np.array([12.99340585, 12.50360028, 12.01378982, 11.40152281, 10.91172647, 10.42202481,  9.93268775,  9.44460172,  8.96057824,  8.48924775, 8.05553588,  7.72107812,  7.58889922])   #addition
            env_loglbd_basse_cb1 = np.array([ -8.92185629,  -9.08561564,  -9.24959912,  -9.45502133, -9.61987743,  -9.78536721,  -9.95165993, -10.11874816, -10.28599223, -10.45059216, -10.60375096, -10.72294597, -10.77030007])   #addition
            env_loglbd_haute_cb3F = np.array([12.99343617, 12.50365356, 12.01388347, 11.40171231, 10.91205951, 10.42261004,  9.93371579,  9.44640577,  8.96373408,  8.49471586, 8.06475136,  7.73553125,  7.6085424 ])   #addition
            env_loglbd_basse_cb3F = np.array([0.87614334, 0.86280792, 0.84474879, 0.81402002, 0.78214142, 0.74399865, 0.70122384, 0.65738833, 0.61861424, 0.59446484, 0.59964507, 0.6562899 , 0.79083947])   #addition
            env_loglbd_haute_cb5 = np.array([13.01296498, 12.52317085, 12.03338046, 11.42116087, 10.9314349 , 10.44185498,  9.95272592,  9.46498659,  8.98151242,  8.51095641, 8.07802122,  7.74341   ,  7.6085424 ])   #addition
            env_loglbd_basse_cb5 = np.array([6.00000043, 5.51020542, 5.02041231, 4.40818023, 3.91841975, 3.42873329, 2.94292516, 2.49834507, 2.07342691, 1.67624622, 1.31818196, 1.01583413, 0.79083947])   #addition
            # Initialisation of the interpolation with the envelopes
            cb3Bmin, cb3Bmax = interp1d(env_logqb3B, env_loglbd_basse_cb3B), interp1d(env_logqb3B, env_loglbd_haute_cb3B) #addition
            cb1min, cb1max = interp1d(env_logq, env_loglbd_basse_cb1), interp1d(env_logq, env_loglbd_haute_cb1) #addition
            cb3Fmin, cb3Fmax = interp1d(env_logq, env_loglbd_basse_cb3F), interp1d(env_logq, env_loglbd_haute_cb3F) #addition
            cb5min, cb5max = interp1d(env_logq, env_loglbd_basse_cb5), interp1d(env_logq, env_loglbd_haute_cb5) #addition
            interp = {'cb3B':(np.float64(cb3Bmin(np.log10(self.q))), np.float64(cb3Bmax(np.log10(self.q)))), 'cb1':(np.float64(cb1min(np.log10(self.q))), np.float64(cb1max(np.log10(self.q)))), 'cb3F':(np.float64(cb3Fmin(np.log10(self.q))), np.float64(cb3Fmax(np.log10(self.q)))), 'cb5':(np.float64(cb5min(np.log10(self.q))), np.float64(cb5max(np.log10(self.q))))}
        elif self.topo == 'interm':
            # Envelopes data for the 2 cusps compute with brentq
            env_logq = np.log10([1.00000000e-06, 1.32571137e-06, 1.75751062e-06, 2.32995181e-06, 3.08884360e-06, 4.09491506e-06, 5.42867544e-06, 7.19685673e-06, 9.54095476e-06, 1.26485522e-05, 1.67683294e-05, 2.94705170e-05, 3.90693994e-05, 5.17947468e-05, 6.86648845e-05, 9.10298178e-05, 1.20679264e-04, 1.59985872e-04, 2.12095089e-04, 2.81176870e-04, 3.72759372e-04, 4.94171336e-04, 6.55128557e-04, 8.68511374e-04, 1.15139540e-03, 1.52641797e-03, 2.02358965e-03, 2.68269580e-03, 3.55648031e-03, 4.71486636e-03, 6.25055193e-03, 8.28642773e-03, 1.09854114e-02, 1.45634848e-02, 1.93069773e-02, 2.55954792e-02, 3.39322177e-02, 4.49843267e-02, 5.96362332e-02, 7.90604321e-02, 1.04811313e-01, 1.38949549e-01, 1.84206997e-01, 2.44205309e-01, 3.23745754e-01, 4.29193426e-01, 5.68986603e-01, 7.54312006e-01, 1.00000000e+00])   #addition
            env_loglbd_haute_cb3 = np.array([2.07614334, 2.07319448, 2.06999984, 2.06654317, 2.06280792, 2.0587774 , 2.05443492, 2.04976404, 2.04474879, 2.03937395, 2.0336254 , 2.02095818, 2.01402002, 2.00667006, 1.99890559, 1.99072763, 1.98214142, 1.97315701, 1.96378981, 1.9540611 , 1.94399865, 1.93363723, 1.92301919, 1.91219501, 1.90122384, 1.89017405, 1.87912387, 1.86816201, 1.85738833, 1.8469147 , 1.83686595, 1.82738099, 1.81861424, 1.81073728, 1.80394095, 1.7984378 , 1.79446484, 1.79228672, 1.79219887, 1.79453028, 1.79964507, 1.80794182, 1.81984903, 1.83581486, 1.8562899 , 1.88170205, 1.91242545, 1.94874731, 1.99083947])   #addition
            env_loglbd_basse_cb3 = np.array([4.34294265e-07, 5.75748749e-07, 7.63276496e-07, 1.01188404e-06, 1.34146566e-06, 1.77839537e-06, 2.35763739e-06, 3.12554392e-06, 4.14356424e-06, 5.49316167e-06, 7.28233186e-06, 1.27986943e-05, 1.69672931e-05, 2.24935902e-05, 2.98197567e-05, 3.95319483e-05, 5.24071763e-05, 6.94754240e-05, 9.21019599e-05, 1.22096398e-04, 1.61857173e-04, 2.14562873e-04, 2.84425560e-04, 3.77025995e-04, 4.99757015e-04, 6.62409472e-04, 8.77945816e-04, 1.16351999e-03, 1.54181967e-03, 2.04282839e-03, 2.70613159e-03, 3.58392131e-03, 4.74488874e-03, 6.27922754e-03, 8.30499702e-03, 1.09760978e-02, 1.44920683e-02, 1.91097767e-02, 2.51568001e-02, 3.30457678e-02, 4.32881128e-02, 5.65044871e-02, 7.34276228e-02, 9.48920503e-02, 1.21804580e-01, 1.55091010e-01, 1.95619235e-01, 2.44106836e-01, 3.01029996e-01])   #addition
            env_loglbd_haute_cb5 = np.array([7.91284938, 7.79136077, 7.66996653, 7.54867589, 7.42749899, 7.30644696, 7.185532  , 7.0647675 , 6.94416814, 6.82375001, 6.70353077, 6.46376812, 6.34426912, 6.22505818, 6.10616311, 5.98761439, 5.86944535, 5.75169242, 5.63439545, 5.51759794, 5.40134744, 5.28569578, 5.17069951, 5.05642027, 4.94292516, 4.8302872 , 4.71858579, 4.6079072 , 4.49834507, 4.39000103, 4.2829853 , 4.17741738, 4.07342691, 3.9711546 , 3.8707534 , 3.77238992, 3.67624622, 3.58252193, 3.49143688, 3.40323394, 3.31818196, 3.23657801, 3.15874798, 3.08504359, 3.01583413, 2.95149056, 2.8923614 , 2.83874166, 2.79083947])   #addition
            env_loglbd_basse_cb5 = np.array([6.00000043, 5.8775516 , 5.7551028 , 5.63265407, 5.51020542, 5.38775688, 5.26530848, 5.14286027, 5.02041231, 4.89796468, 4.77551749, 4.53062504, 4.40818023, 4.28573678, 4.16329513, 4.04085586, 3.91841975, 3.79598784, 3.67356149, 3.5511425 , 3.42873329, 3.30633701, 3.18395789, 3.06160152, 2.93927527, 2.81698894, 2.6947555 , 2.57259209, 2.45052141, 2.32857344, 2.20678776, 2.08521657, 1.96392856, 1.84301392, 1.72259071, 1.60281283, 1.48387982, 1.36604855, 1.2496466 , 1.13508658, 1.02287995, 0.91364734, 0.8081215 , 0.70713695, 0.6116005 , 0.52243795, 0.44051719, 0.36655582, 0.30103])   #addition
            # Initialisation of the interpolation with the envelopes
            cb3min, cb3max = interp1d(env_logq, env_loglbd_basse_cb3), interp1d(env_logq, env_loglbd_haute_cb3) #addition
            cb5min, cb5max = interp1d(env_logq, env_loglbd_basse_cb5), interp1d(env_logq, env_loglbd_haute_cb5) #addition
            interp = {'cb3':(np.float64(cb3min(np.log10(self.q))), np.float64(cb3max(np.log10(self.q)))), 'cb5':(np.float64(cb5min(np.log10(self.q))), np.float64(cb5max(np.log10(self.q))))}
        elif self.topo == 'wide':
            # Envelopes data for the 2 cusps compute with brentq
            env_logq = np.log10([1.00000000e-06, 1.32571137e-06, 1.75751062e-06, 2.32995181e-06, 3.08884360e-06, 4.09491506e-06, 5.42867544e-06, 7.19685673e-06, 9.54095476e-06, 1.26485522e-05, 1.67683294e-05, 2.94705170e-05, 3.90693994e-05, 5.17947468e-05, 6.86648845e-05, 9.10298178e-05, 1.20679264e-04, 1.59985872e-04, 2.12095089e-04, 2.81176870e-04, 3.72759372e-04, 4.94171336e-04, 6.55128557e-04, 8.68511374e-04, 1.15139540e-03, 1.52641797e-03, 2.02358965e-03, 2.68269580e-03, 3.55648031e-03, 4.71486636e-03, 6.25055193e-03, 8.28642773e-03, 1.09854114e-02, 1.45634848e-02, 1.93069773e-02, 2.55954792e-02, 3.39322177e-02, 4.49843267e-02, 5.96362332e-02, 7.90604321e-02, 1.04811313e-01, 1.38949549e-01, 1.84206997e-01, 2.44205309e-01, 3.23745754e-01, 4.29193426e-01, 5.68986603e-01, 7.54312006e-01, 1.00000000e+00])   #addition
            env_loglbd_haute_cb3 = np.array([0.08034022, 0.08037904, 0.08042241, 0.08047094, 0.0805253 , 0.0805863 , 0.08065486, 0.08073207, 0.08081916, 0.08091759, 0.08102907, 0.08129952, 0.08146359, 0.08165109, 0.08186586, 0.0821125 , 0.08239642, 0.08272412, 0.0831033 , 0.08354318, 0.08405481, 0.0846514 , 0.08534884, 0.08616624, 0.08712662, 0.08825776, 0.08959329, 0.09117398, 0.0930494 , 0.09527999, 0.09793967, 0.10111916, 0.10493015, 0.10951055, 0.11503107, 0.12170336, 0.1297898 , 0.13961513, 0.1515795 , 0.16617224, 0.18398473, 0.20571946, 0.23219114, 0.26431411, 0.30307002, 0.34945112, 0.40437973, 0.46861176, 0.54264165])   #addition
            env_loglbd_basse_cb3 = np.array([4.34294265e-07, 5.75748749e-07, 7.63276496e-07, 1.01188404e-06, 1.34146566e-06, 1.77839537e-06, 2.35763739e-06, 3.12554392e-06, 4.14356424e-06, 5.49316167e-06, 7.28233186e-06, 1.27986943e-05, 1.69672931e-05, 2.24935902e-05, 2.98197567e-05, 3.95319483e-05, 5.24071763e-05, 6.94754240e-05, 9.21019599e-05, 1.22096398e-04, 1.61857173e-04, 2.14562873e-04, 2.84425560e-04, 3.77025995e-04, 4.99757015e-04, 6.62409472e-04, 8.77945816e-04, 1.16351999e-03, 1.54181967e-03, 2.04282839e-03, 2.70613159e-03, 3.58392131e-03, 4.74488874e-03, 6.27922754e-03, 8.30499702e-03, 1.09760978e-02, 1.44920683e-02, 1.91097767e-02, 2.51568001e-02, 3.30457678e-02, 4.32881128e-02, 5.65044871e-02, 7.34276228e-02, 9.48920503e-02, 1.21804580e-01, 1.55091010e-01, 1.95619235e-01, 2.44106836e-01, 3.01029996e-01])   #addition
            env_loglbd_haute_cb5 = np.array([7.3837899 , 7.25946526, 7.13496117, 7.01026106, 6.88534694, 6.76019931, 6.6347971 , 6.50911753, 6.38313606, 6.25682626, 6.13015983, 5.87563372, 5.74770735, 5.61929093, 5.49034616, 5.36083293, 5.23070952, 5.09993288, 4.96845903, 4.83624357, 4.70324237, 4.56941243, 4.43471292, 4.29910656, 4.16256121, 4.02505189, 3.88656308, 3.74709157, 3.60664975, 3.46526944, 3.32300642, 3.17994548, 3.0362064 , 2.89195058, 2.74738862, 2.60278881, 2.45848655, 2.31489461, 2.17251389, 2.03194391, 1.89389159, 1.75917573, 1.62872345, 1.50355324, 1.38473924, 1.27335258, 1.17038044, 1.07663102, 0.99264165])   #addition
            env_loglbd_basse_cb5 = np.array([6.00000043, 5.8775516 , 5.7551028 , 5.63265407, 5.51020542, 5.38775688, 5.26530848, 5.14286027, 5.02041231, 4.89796468, 4.77551749, 4.53062504, 4.40818023, 4.28573678, 4.16329513, 4.04085586, 3.91841975, 3.79598784, 3.67356149, 3.5511425 , 3.42873329, 3.30633701, 3.18395789, 3.06160152, 2.93927527, 2.81698894, 2.6947555 , 2.57259209, 2.45052141, 2.32857344, 2.20678776, 2.08521657, 1.96392856, 1.84301392, 1.72259071, 1.60281283, 1.48387982, 1.36604855, 1.2496466 , 1.13508658, 1.02287995, 0.91364734, 0.8081215 , 0.70713695, 0.6116005 , 0.52243795, 0.44051719, 0.36655582, 0.30103   ])   #addition
            # Initialisation of the interpolation with the envelopes
            cb3min, cb3max = interp1d(env_logq, env_loglbd_basse_cb3), interp1d(env_logq, env_loglbd_haute_cb3) #addition
            cb5min, cb5max = interp1d(env_logq, env_loglbd_basse_cb5), interp1d(env_logq, env_loglbd_haute_cb5) #addition
            interp = {'cb3':(np.float64(cb3min(np.log10(self.q))), np.float64(cb3max(np.log10(self.q)))), 'cb5':(np.float64(cb5min(np.log10(self.q))), np.float64(cb5max(np.log10(self.q))))}
        return interp


    def _sortbnccu(self, lbd):
        """Solve normalized cusp curve equation and sort branches"""
        if lbd <= mp.mpf(0.):
            # solve cusp equation for saddle point (lbd = 0)
            a3 = mp.mpf(1.)
            a2 = mp.mpf(3.) / mp.mpf(1. + self.q)
            a1 = mp.mpf(3.) / mp.mpf(1. + self.q)
            a0 = mp.mpf(1.) / mp.mpf(1. + self.q)
            zn = np.array(mp.polyroots([a3, a2, a1, a0], maxsteps=60, extraprec=self.extraprec), dtype=np.complex128)
            arg = np.argsort(np.imag(zn))
            sort = zn[arg]
            b2 = complex(sort[0])
            b4 = b2
            b5 = complex(sort[1])
            b6 = b5
            b1 = complex(sort[2])
            b3 = b1
            # bn, sn
            bn = np.array([b1, b2, b3, b4, b5, b6])
            sn = np.sqrt(np.abs(1. / (1. + self.q) * (1. / bn**2 + self.q / (bn + 1.)**2)))
        elif lbd == mp.mpf(1.):
            # solve cusp equation for 4 non-∞ branches (lbd = 1)
            L = mp.mpf(1.) / mp.mpf(1. + self.q)
            a4 = mp.mpf(3. * (5. * (1. - L) + 2. * (1. - 3. * L) * self.q  - L * self.q**2))
            a3 = mp.mpf(2. * (10. * (1. - L) + (1. - 6. * L) * self.q))
            a2 = mp.mpf(3. * (5. * (1. - L) - L * self.q))
            a1 = mp.mpf(6. * (1. - L))
            a0 = mp.mpf(1. - L)
            zn = np.array(mp.polyroots([a4, a3, a2, a1, a0], maxsteps=60, extraprec=self.extraprec), dtype=np.complex128)
            # b1
            arg = np.argmax(np.imag(zn))
            b1 = complex(zn[arg])
            zn = np.delete(zn, arg)
            # b2
            arg = np.argmin(np.imag(zn))
            b2 = complex(zn[arg])
            zn = np.delete(zn, arg)
            # b3, b4
            b3 = np.inf
            b4 = -np.inf
            # b5, b6
            arg = np.argsort(np.real(zn))
            sort = zn[arg]
            b5, b6 = complex(sort[0]), complex(sort[1])
            # bn, sn
            bn = np.array([b1, b2, b3, b4, b5, b6])
            sn = np.zeros(6, dtype=np.float_) # s -> 0. for zn -> ∞
            sn[[0, 1, 4, 5]] = np.sqrt(np.abs(1. / (1. + self.q) * (1. / bn[[0, 1, 4, 5]]**2 + self.q / (bn[[0, 1, 4, 5]] + 1.)**2)))
        elif lbd == mp.mpf(1. + self.q):
            # solve cusp equation for primary lens position (lbd = 1+q)
            a4 = mp.mpf(- self.q * (1. + self.q)**2)
            a3 = mp.mpf(-6. * self.q * (1. + self.q))
            a2 = mp.mpf(-3. * self.q * (self.q + 4.))
            a1 = mp.mpf(-10. * self.q)
            a0 = mp.mpf(-3. * self.q)
            zn = np.array(mp.polyroots([a4, a3, a2, a1, a0], maxsteps=60, extraprec=self.extraprec), dtype=np.complex128)
            # b1
            arg = np.argmax(np.imag(zn))
            b1 = complex(zn[arg])
            zn = np.delete(zn, arg)
            # b2
            arg = np.argmin(np.imag(zn))
            b2 = complex(zn[arg])
            zn = np.delete(zn, arg)
            # b4, b5
            arg = np.argsort(np.real(zn))
            sort = zn[arg]
            b4, b5 = complex(sort[0]), complex(sort[1])
            # b3, b6
            b3 = 0j
            b6 = 0j
            # bn, sn
            bn = np.array([b1, b2, b3, b4, b5, b6])
            sn = np.zeros(6, dtype=np.float_)
            sn[[2, 5]] = np.inf # s -> ∞ for zn -> 0.
            sn[[0, 1, 3, 4]] = np.sqrt(np.abs(1. / (1. + self.q) * (1. / bn[[0, 1, 3, 4]]**2 + self.q / (bn[[0, 1, 3, 4]] + 1.)**2)))
        elif lbd == mp.mpf(1. + 1. / self.q):
            # solve cusp equation for secondary lens position (lbd = 1+1/q)
            a4 = mp.mpf(-2. - 1. / self.q - self.q)
            a3= mp.mpf(6. * (1. + self.q))
            a2 = mp.mpf(-3. * (1. + 4. * self.q))
            a1 = mp.mpf(10. * self.q)
            a0 = mp.mpf(-3. * self.q)
            Zn = np.array(mp.polyroots([a4, a3, a2, a1, a0], maxsteps=60, extraprec=self.extraprec), dtype=np.complex128)   # Zn = zn + 1.
            # b1, b3
            arg = np.where(np.imag(Zn) > 0.)
            B = Zn[arg]
            arg = np.argmin(np.real(B))
            b1 = complex(B[arg]) - 1. # Zn = zn + 1.
            arg = np.argmax(np.real(B))
            b3 = complex(B[arg]) - 1. # Zn = zn + 1.
            # b2, b6
            b2 = np.conj(b1)
            b6 = np.conj(b3)
            # b4, b5
            b4 = 0j - 1. # Zn = zn + 1.
            b5 = 0j - 1. # Zn = zn + 1.
            # bn, sn
            bn = np.array([b1, b2, b3, b4, b5, b6])
            sn = np.zeros(6, dtype=np.float_)
            sn[[3, 4]] = np.inf # s -> ∞ for zn -> -1.
            sn[[0, 1, 2, 5]] = np.sqrt(np.abs(1. / (1. + self.q) * (1. / bn[[0, 1, 2, 5]]**2 + self.q / (bn[[0, 1, 2, 5]] + 1.)**2)))
        elif lbd == mp.mpf(np.inf):
            # solve cusp equation for maximum of Jacobian (lbd = ∞)
            # b1, b3, b5
            b1 = -1. / (1. + self.q) * np.complex(1., -np.sqrt(self.q))
            b3, b5 = b1, b1
            # b2, b4, b6
            b2 = np.conj(b1)
            b4, b6 = b2, b2
            # bn, sn
            bn = np.array([b1, b2, b3, b4, b5, b6])
            sn = np.zeros(6, dtype=np.float_)
        else:
            # solve cusp equation for intervals with high precision
            L = mp.mpf(lbd) / mp.mpf(1. + self.q)
            a6 = mp.mpf(1. - L + (2. - 3. * L) * self.q + (1. - 3. * L)* self.q**2 - L * self.q**3)
            a5 = mp.mpf(6. * (1. - L + (1. -2. * L) * self.q - L * self.q**2))
            a4 = mp.mpf(3. * (5. * (1. - L) + 2. * (1. - 3. * L) * self.q - L * self.q**2))
            a3 = mp.mpf(2. * (10. * (1. - L) + (1. - 6. * L) * self.q))
            a2 = mp.mpf(3. * (5. * (1. - L) - L * self.q))
            a1 = mp.mpf(6. * (1. - L))
            a0 = mp.mpf(1. - L)
            zn = np.array(mp.polyroots([a6, a5, a4, a3, a2, a1, a0], maxsteps=60, extraprec=self.extraprec), dtype=np.complex128)
            # Values comparisons between lbd (high precision) and high precision numbers
            if mp.mpf(0.) < lbd < mp.mpf(1.):
                # 0 < lbd < 1 (saddle -> ∞)
                # b3
                # arg = np.argmax([mp.im(i) for i in zn])
                arg = np.argmax(np.imag(zn))
                b3 = complex(zn[arg])
                zn = np.delete(zn, arg)
                # b4
                arg = np.argmin(np.imag(zn))
                b4 = complex(zn[arg])
                zn = np.delete(zn, arg)
                # b1
                arg = np.argmax(np.imag(zn))
                b1 = complex(zn[arg])
                zn = np.delete(zn, arg)
                # b2
                arg = np.argmin(np.imag(zn))
                b2 = complex(zn[arg])
                zn = np.delete(zn, arg)
                # b5, b6
                arg = np.argsort(np.real(zn))
                sort = zn[arg]
                b5, b6 = complex(sort[0]), complex(sort[1])
            elif mp.mpf(1.) < lbd < mp.mpf(1. + self.q):
                # 1 < lbd < 1/mu_1 = 1+q (∞ -> left lens pos)
                # b1
                arg = np.argmax(np.imag(zn))
                b1 = complex(zn[arg])
                zn = np.delete(zn, arg)
                # b2
                arg = np.argmin(np.imag(zn))
                b2 = complex(zn[arg])
                zn = np.delete(zn, arg)
                # b3, b4, b5, b6
                arg = np.argsort(np.real(zn))
                sort = zn[arg]
                b4, b5, b6, b3 = complex(sort[0]), complex(sort[1]), complex(sort[2]), complex(sort[3])
            elif mp.mpf(1. + self.q) < lbd < mp.mpf(1.) + mp.mpf(1.) / mp.mpf(self.q):
                # 1/mu_1 = 1+q < lbd < 1/mu_2 = 1+1/q (lef lens -> right lens)
                # b4, b5
                arg = np.argsort(np.imag(zn))
                sort = zn[arg]
                B = sort[[2, 3]]
                zn = np.delete(zn, arg[[2, 3]])
                arg = np.argsort(np.real(B))
                sort = sort[arg]
                b4, b5 = complex(B[0]), complex(B[1])
                # b1, b3
                arg = np.where(np.imag(zn) > 0.)
                B = zn[arg]
                arg = np.argsort(np.real(B))
                sort = B[arg]
                b1, b3 = complex(sort[0]), complex(sort[1])
                # b2, b6
                b2, b6 = np.conj(b1), np.conj(b3)
            elif mp.mpf(1.) + mp.mpf(1.) / mp.mpf(self.q) < lbd:
                # 1/mu_2 = 1+1/q < lbd (right lens -> max jacobian)
                # b1, b3, b5
                arg = np.where(np.imag(zn) > 0.)
                B = zn[arg]
                arg = np.argmax(np.real(B))
                b3 = complex(B[arg])
                B = np.delete(B, arg)
                arg = np.argsort(np.imag(B))
                C = B[arg]
                # Using a patch to lead the cumpute out of an unphysical case (images at infinite distance)
                if np.size(C) == 2:
                    b5, b1 = complex(C[0]), complex(C[1])
                else:
                    b5 = complex(C[0])
                    b1 = np.complex(-0.999999, 0.0000001)
                # b2, b4, b6
                b2, b4, b6 = np.conj(b1), np.conj(b5), np.conj(b3)
            # bn, sn
            bn = np.array([b1, b2, b3, b4, b5, b6])
            sn = np.sqrt(np.abs(1. / (1. + self.q) * (1. / bn**2 + self.q / (bn + 1.)**2)))
        return bn, sn

    def pltcrit(self, axis=None, figname=None):
        """Plot critical curves

            Parameters
            ----------
            axis : float 1-D array, optional
                Plot limits. Usage: axis=[xmin, xmax, ymin, ymax].
                Default is: automatic axis (equal units in x and y).
            figname : str, optional
                Name of output figure. Example: figname='critcurves.pdf'.
                Default is: None.

            Returns
            -------
            out : display the critical curves
                Default is: display plot on screen with pylab.show().
                If figname is specified, write file instead of display.
            ValueError : an error is raised if N=None

            Examples
            --------
            >>> cc = Caustics(1.4, 0.1, N=300)
            >>> cc.pltcrit()
            >>> cc.pltcrit(axis=[-4, 4, -1, 2], figname='critcurves.pdf')

            >>> cc = Caustics(0.3, 0.0001, N=400, cusp=True)
            >>> cc.pltcaus()
            """
        if self._N:
            plt.close('all')
            fig, CRI = plt.subplots(1, figsize=(6,6))
            plt.subplots_adjust(left=0.14, bottom=0.14, right=0.94, top=0.94, wspace=None, hspace=None)
            CRI.set_aspect('equal')
            CRI.set_title('Critical curves')
            CRI.set_xlabel(self._crilabels[0])
            CRI.set_ylabel(self._crilabels[1])
            if axis:
                axis = np.array(axis)
                CRI.set_xlim([axis[0], axis[1]])
                CRI.set_ylim([axis[2], axis[3]])
            CRI.scatter(np.real(self.zc), np.imag(self.zc), marker='o', c='red', s=0.5)
            # plot lens positions, ref. CM
            L1, L2 = -self.s + self._AC2CM, self._AC2CM
            CRI.scatter([L1, L2], [0., 0.], marker='+', c='midnightblue', s=80)
            # save & plot
            if figname:
                figname = os.getcwd() + '/' + figname
                plt.savefig(figname)
            else: plt.show()
        else: raise ValueError("no curve to plot (input N=None)")

    def pltcaus(self, axis=None, figname=None):
        """Plot caustics and cusps

            Parameters
            ----------
            axis : float 1-D array, optional
                Plot limits. Usage: axis=[xmin, xmax, ymin, ymax].
                Default is: automatic axis (equal units in x and y).
            figname : str, optional
                Name of output figure. Example: figname='caustics.pdf'.
                Default is: None.

            Returns
            -------
            out : display the caustics and cusps
                Default is: display plot on screen with pylab.show().
                If cusp=True, display circles around cusps.
                If figname is specified, write file instead of display.
            ValueError : an error is raised if N=None

            Examples
            --------
            >>> cc = Caustics(1.4, 0.1, N=300, cusp=True)
            >>> cc.pltcaus()
            >>> cc.pltcaus(axis=[-4, 4, -1, 2], figname='caustics.pdf')
            """
        if self._N:
            plt.close('all')
            fig, CAU = plt.subplots(1, figsize=(6,6))
            plt.subplots_adjust(left=0.14, bottom=0.14, right=0.94, top=0.94, wspace=None, hspace=None)
            CAU.set_aspect('equal')
            CAU.set_title('Caustics')
            CAU.set_xlabel(self._caulabels[0])
            CAU.set_ylabel(self._caulabels[1])
            if axis:
                axis = np.array(axis)
                CAU.set_xlim([axis[0], axis[1]])
                CAU.set_ylim([axis[2], axis[3]])
            # plot caustics
            CAU.scatter(np.real(self.zetac), np.imag(self.zetac), marker='o', c='red', s=0.5)
            # plot cusps
            if self._cusp:
                [CAU.scatter(np.real(self.cusps[key]), np.imag(self.cusps[key]),  marker='o', facecolors='none', edgecolors='darkred', s=60) for key in self.cusps]
            # plot lens positions, ref. CM
            L1, L2 = -self.s + self._AC2CM, self._AC2CM
            CAU.scatter([L1, L2], [0., 0.], marker='+', c='midnightblue', s=80)
            # save & plot
            if figname:
                figname = os.getcwd() + '/' + figname
                plt.savefig(figname)
            else: plt.show()
        else: raise ValueError("no curve to plot (input N=None)")


if __name__ == '__main__':
    cc = Caustics(2., 1., N=200, cusp=True)
    # cc.pltcrit()
    # cc.pltcaus()
    #print (cc.topo)
    #for key in cc.cusps:
        #print key, cc.cusps[key]
