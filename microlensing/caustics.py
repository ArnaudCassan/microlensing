# -*- coding: utf-8 -*-
"""Binary lens critical curves, caustics and cusps"""

# Copyright (c) 2017-2019 Arnaud Cassan
# Distributed under the terms of the MIT license

# This module is part of the microlensing suite:
#       https://github.com/ArnaudCassan/microlensing

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import copy
import scipy.optimize as opt
import warnings

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
        >>> print(cc.topo)
        'interm'
        
        >>> cc = Caustics(0.3, 0.0001, N=400, cusp=True)
        >>> cc.pltcaus()
        >>> for key in cc.cusps:
                print(key, cc.cusps[key])
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
        self._cusp = cusp
        self.cusps = dict()
        self.zc = None
        self.zetac = None
        self._zcu = None
        self._AC2CM = s * q / (1. + q)
        self._caulabels = [r'$\xi \:[\theta_{\rm E}]$', r'$\eta \:[\theta_{\rm E}]$']
        self._crilabels = [r'$x \:[\theta_{\rm E}]$', r'$y \:[\theta_{\rm E}]$']
        self._findtopo()
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
        # find topology
        if self.s < c2i: self.topo = 'close'
        if c2i <= self.s <= i2w: self.topo = 'interm'
        if i2w < self.s: self.topo = 'wide'

    def _getcusps(self):
        """Get cusps coordinates"""
        # compute (Ox) roots
        witt = [1., 2. * self.s, self.s**2 - 1., -2. * self.s / (1. + self.q), -(self.s**2 / (1. + self.q))]
        zc = np.roots(witt)
        # topology: close
        if self.topo == 'close':
            maxiter = 100
            xtol = 1e-16
            # (Ox)
#            zc = copy(self.zc[0] - self._AC2CM)
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
            lbd = opt.brentq(self._fpoly, 0., 1., args=(2), xtol=xtol, maxiter=maxiter)
            bn, sn = self._sortbnccu(lbd)
            ccB = complex(bn[2]) * self.s
            # D -> conj(B)
            ccD = np.conj(ccB)
            ## secondary caustics
            lbdmax = 1e8
            epsc = 1e-2
            phi = np.array([0., 2. * np.pi / 3., 4. * np.pi / 3.])
            z0 = - 1. / (1. + self.q) * (1. - 1j * np.sqrt(self.q))
            bn, sn = self._sortbnccu(0.)
            zsadd = complex(bn[0])
            beta = - np.pi / 6. + 4. / 3. * np.angle(np.sqrt(self.q) + 1j) + phi ## -4/3 -> +4/3
            T = - np.exp(1j * beta)
            # E, b1, [0, ∞] -> poly
            try:
                lbd = opt.brentq(self._fpoly, 0., lbdmax, args=(0), xtol=xtol, maxiter=maxiter)
                bn, sn = self._sortbnccu(lbd)
                ccE = complex(bn[0])
            except: ccE = np.nan
            if (np.abs(np.sqrt(np.abs(1. / (1. + self.q) * (1. / ccE**2 + self.q / (ccE + 1.)**2))) - self.s) / self.s > epsc) or np.isnan(ccE):
                testr = np.array([np.abs(zsadd - z0), - np.imag(z0) / np.imag(T[2])])
                arg = np.where(testr > 0.)
                rmax = np.min(testr[arg])
                r = opt.brentq(self._fapprox, 0., rmax, args=(z0, T[2]), xtol=xtol, maxiter=maxiter)
                ccE = z0 + r * T[2]
            ccE = ccE * self.s
            # H -> conj(E)
            ccH = np.conj(ccE)
            # F, b3, [1+q, ∞] -> poly / approx
            try:
                lbd = opt.brentq(self._fpoly, 1. + self.q, lbdmax, args=(2), xtol=xtol, maxiter=maxiter)
                bn, sn = self._sortbnccu(lbd)
                ccF = complex(bn[2])
            except: ccF = np.nan
            if (np.abs(np.sqrt(np.abs(1. / (1. + self.q) * (1. / ccF**2 + self.q / (ccF + 1.)**2))) - self.s)  / self.s > epsc) or np.isnan(ccF):
                testr = np.array([np.abs(zsadd - z0), - np.imag(z0) / np.imag(T[1])])
                arg = np.where(testr > 0.)
                rmax = np.min(testr[arg])
                r = opt.brentq(self._fapprox, 0., rmax, args=(z0, T[1]), xtol=xtol, maxiter=maxiter)
                ccF = z0 + r * T[1]
            ccF = ccF * self.s
            # I -> conj(F)
            ccI = np.conj(ccF)
            # G, b5, [1+1/q, ∞] -> poly / approx
            try:
                lbd = opt.brentq(self._fpoly, 1. + 1. / self.q, lbdmax, args=(4), xtol=xtol, maxiter=maxiter)
                bn, sn = self._sortbnccu(lbd)
                ccG = complex(bn[4])
            except: ccG = np.nan
            if (np.abs(np.sqrt(np.abs(1. / (1. + self.q) * (1. / ccG**2 + self.q / (ccG + 1.)**2))) - self.s)  / self.s > epsc) or np.isnan(ccG):
                testr = np.array([np.abs(zsadd - z0), - np.imag(z0) / np.imag(T[0])])
                arg = np.where(testr > 0.)
                rmax = np.min(testr[arg])
                r = opt.brentq(self._fapprox, 0., rmax, args=(z0, T[0]), xtol=xtol, maxiter=maxiter)
                ccG = z0 + r * T[0]
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
            maxiter = 100
            xtol = 1e-16
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
            lbdmax = 1e4
            try: lbd = opt.brentq(self._fpoly, 1. + self.q, lbdmax, args=(2), xtol=xtol, maxiter=maxiter)
            except: lbd = 1e3 ## TEST : plutot pas mal
            bn, sn = self._sortbnccu(lbd)
            ccB = complex(bn[2]) * self.s
            # F -> conj(B)
            ccF = np.conj(ccB)
            # C, b5, [1+1/q, ∞]
            z0 = - 1. / (1. + self.q) * (1. - 1j * np.sqrt(self.q))
            beta = - np.pi / 6. + 4. / 3. * np.angle(np.sqrt(self.q) + 1j) ## -4/3 -> +4/3
            T = - np.exp(1j * beta)
            lbdmax = 5e7
            epsc = 1e-2
            try:
                lbd = opt.brentq(self._fpoly, 1. + 1. / self.q, lbdmax, args=(4), xtol=xtol, maxiter=maxiter)
                bn, sn = self._sortbnccu(lbd)
                ccC = complex(bn[4])
            except: ccC = np.nan
            if (np.abs(np.sqrt(np.abs(1. / (1. + self.q) * (1. / ccC**2 + self.q / (ccC + 1.)**2))) - self.s)  / self.s > epsc) or np.isnan(ccC):
                rmax = np.abs(z0 + 1.)
                r = opt.brentq(self._fapprox, 0., rmax, args=(z0, T), xtol=xtol, maxiter=maxiter)
                ccC = z0 + r * T
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
            maxiter = 100
            xtol = 1e-16
            epsc = 1e-2
            lbdmax = 1e8
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
            lbd = opt.brentq(self._fpoly, 1. + self.q, lbdmax, args=(2), xtol=xtol, maxiter=maxiter)
            bn, sn = self._sortbnccu(lbd)
            ccB = complex(bn[2]) * self.s
            # D -> conj(B)
            ccD = np.conj(ccB)
            ## secondary caustic
            # E, b5,  [0, 1+1/q] -> Witt
            ccE = complex(np.real(sort[1]))
            # G, b4, [1, 1+1/q] -> Witt
            ccG = complex(np.real(sort[0]))
            # F, b5, [1+1/q, ∞] -> poly
            z0 = - 1. / (1. + self.q) * (1. - 1j * np.sqrt(self.q))
            beta = - np.pi / 6. + 4. / 3. * np.angle(np.sqrt(self.q) + 1j) ## -4/3 -> +4/3
            T = - np.exp(1j * beta)
            try:
                lbd = opt.brentq(self._fpoly, 1. + 1. / self.q, lbdmax, args=(4), xtol=xtol, maxiter=maxiter)
                bn, sn = self._sortbnccu(lbd)
                ccF = complex(bn[4])
            except: ccF = np.nan
            if (np.abs(np.sqrt(np.abs(1. / (1. + self.q) * (1. / ccF**2 + self.q / (ccF + 1.)**2))) - self.s)  / self.s > epsc) or np.isnan(ccF):
                rmax = np.abs(z0 + 1.)
                r = opt.brentq(self._fapprox, 0., rmax, args=(z0, T), xtol=xtol, maxiter=maxiter)
                ccF = z0 + r * T
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

    def _fpoly(self, lbd, br):
        """Sub-routine of _getcusps"""
        bn, sn = self._sortbnccu(lbd)
        return sn[br] - self.s

    def _fapprox(self, r, z0, T):
        """Sub-routine of _getcusps"""
        z = z0 + r * T
        return np.sqrt(np.abs(1. / (1. + self.q) * (1. / z**2 + self.q / (z + 1.)**2))) - self.s

    def _sortbnccu(self, lbd):
        """Solve normalized cusp curve equation and sort branches"""
        if lbd <= 0.:
            # solve cusp equation for saddle point (lbd = 0)
            a3 = 1.
            a2 = 3. / (1. + self.q)
            a1 = 3. / (1. + self.q)
            a0 = 1. / (1. + self.q)
            zn = np.roots([a3, a2, a1 , a0])
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
        elif lbd == 1.:
            # solve cusp equation for 4 non-∞ branches (lbd = 1)
            L = 1. / (1. + self.q)
            a4 = 3. * (5. * (1. - L) + 2. * (1. - 3. * L) * self.q  - L * self.q**2)
            a3 = 2. * (10. * (1. - L) + (1. - 6. * L) * self.q)
            a2 = 3. * (5. * (1. - L) - L * self.q)
            a1 = 6. * (1. - L)
            a0 = 1. - L
            zn = np.roots([a4, a3, a2, a1, a0])
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
        elif lbd == 1. + self.q:
            # solve cusp equation for primary lens position (lbd = 1+q)
            a4 = - self.q * (1. + self.q)**2
            a3 = -6. * self.q * (1. + self.q)
            a2 = -3. * self.q * (self.q + 4.)
            a1 = -10. * self.q
            a0 = -3. * self.q
            zn = np.roots([a4, a3, a2, a1, a0])
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
        elif lbd == 1. + 1. / self.q:
            # solve cusp equation for secondary lens position (lbd = 1+1/q)
            a4 = -2. - 1. / self.q - self.q
            a3= 6. * (1. + self.q)
            a2 = -3. * (1. + 4. * self.q)
            a1 = 10. * self.q
            a0 = -3. * self.q
            Zn = np.roots([a4, a3, a2, a1, a0]) # Zn = zn + 1.
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
        elif lbd == np.inf:
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
            # solve cusp equation for intervals
            L = lbd / (1. + self.q)
            a6 = 1. - L + (2. - 3. * L) * self.q + (1. - 3. * L) * self.q**2 - L * self.q**3
            a5 = 6. * (1. - L + (1. - 2. * L) * self.q - L * self.q**2)
            a4 = 3. * (5. * (1. - L) + 2. * (1. - 3. * L) * self.q  - L * self.q**2)
            a3 = 2. * (10. * (1. - L) + (1. - 6. * L) * self.q)
            a2 = 3. * (5. * (1. - L) - L * self.q)
            a1 = 6. * (1. - L)
            a0 = 1. - L
            zn = np.roots([a6, a5, a4, a3, a2, a1, a0])
            if 0. < lbd < 1.:
                # 0 < lbd < 1 (saddle -> ∞)
                # b3
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
            elif 1. < lbd < 1. + self.q:
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
            elif 1. + self.q < lbd < 1. + 1. / self.q:
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
            elif 1. + 1. / self.q < lbd:
                # 1/mu_2 = 1+1/q < lbd (right lens -> max jacobian)
                # b1, b3, b5
                arg = np.where(np.imag(zn) > 0.)
                B = zn[arg]
                arg = np.argmax(B)
                b3 = complex(B[arg])
                B = np.delete(B, arg)
                arg = np.argsort(np.imag(B))
                C = B[arg]
                b5, b1 = complex(C[0]), complex(C[1])
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
    cc = Caustics(0.3, 0.0001, N=400, cusp=True)
    cc.pltcrit()
    cc.pltcaus()
    print(cc.topo)
    for key in cc.cusps:
        print(key, cc.cusps[key])


