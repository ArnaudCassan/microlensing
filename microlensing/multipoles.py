# -*- coding: utf-8 -*-
"""Multipole expansion of binary-lens finite-source magnification"""

# Copyright (c) 2017-2019 Arnaud Cassan
# Distributed under the terms of the MIT license

# This module is part of the microlensing suite:
#       https://github.com/ArnaudCassan/microlensing

# This module is based on method presented in publication:
#   Cassan, A. (2017), Fast computation of quadrupole and hexadecapole
#       approximations in microlensing with a single point-source
#       evaluation, Mon. Not. R. Astron. 468, 3993.
# Please quote if used for a publication

import numpy as np

def quadrupole(Wk, rho, Gamma):
    """Quadrupole expansion of finite-source magnification
        
        Parameters
        ----------
        Wk : complex 2-D array
            Derivatives of lens equation (cf. puplication).
        rho : float
            Source radius in Theta_E units.
        Gamma : float
            Linear limb-darkening coefficient.
        
        Returns
        -------
        out : float 1D array
            Monopole and quadrupole orders of the expansion.
            
        See Also
        --------
        Cassan, A. (2017), Fast computation of quadrupole and hexadecapole
        approximations in microlensing with a single point-source
        evaluation, Mon. Not. R. Astron. 468, 3993.
        """
    # monopole magnification
    mu0 = 1. / (1. - np.abs(Wk[2])**2)
    
    # order p=1 (monopole)
    a10 = mu0 * (1. + np.conj(Wk[2]))
    a01 = mu0 * 1j * (1. - np.conj(Wk[2]))
    
    # order p=2
    Q20 = Wk[3] * a10 * a10
    Q11 = Wk[3] * a10 * a01
    Q02 = Wk[3] * a01 * a01
    a20 =  _akl(mu0, Wk[2], Q20)
    a11 =  _akl(mu0, Wk[2], Q11)
    a02 =  _akl(mu0, Wk[2], Q02)
    
    # order p=3 (quadrupole)
    Q30 = Wk[3] * 3. * a20 * a10 + Wk[4] * a10 * a10 * a10
    Q21 = Wk[3] * ( 2.*a11*a10 + a01*a20 ) + Wk[4] * a01 * a10 * a10
    Q12 = Wk[3] * ( 2. * a11 * a01 + a10 * a02 ) + Wk[4] * a10 * a01 * a01
    Q03 = Wk[3] * 3. * a02 * a01 + Wk[4] * a01 * a01 * a01
    a30 =  _akl(mu0, Wk[2], Q30)
    a21 =  _akl(mu0, Wk[2], Q21)
    a12 =  _akl(mu0, Wk[2], Q12)
    a03 =  _akl(mu0, Wk[2], Q03)
    
    # compute quadrupolar mu and A
    mu2 = 1./4. * np.imag(a01 * np.conj(a12 + a30) + np.conj(a10) * (a03 + a21) + 2. * a02 * np.conj(a11) + 2. * a11 * np.conj(a20))
    A0 = np.sum(np.abs(mu0))
    A2 = np.sum(np.abs(mu0 + 1. / 2. * mu2 * (1. - 1./5. * Gamma) * rho**2))
    
    return np.array([A0, A2])

def hexadecapole(Wk, rho, Gamma):
    """Hexadecapole expansion of finite-source magnification
        
        Parameters
        ----------
        Wk : complex 2-D array
            Derivatives of lens equation (cf. puplication).
        rho : float
            Source radius in Theta_E units.
        Gamma : float
            Linear limb-darkening coefficient.
        
        Returns
        -------
        out : float 1D array
            Monopole, quadrupole and hexadecapole orders of the expansion.
            
        See Also
        --------
        Cassan, A. (2017), Fast computation of quadrupole and hexadecapole
        approximations in microlensing with a single point-source
        evaluation, Mon. Not. R. Astron. 468, 3993.
        """
    # monopole magnification
    mu0 = 1. / (1. - np.abs(Wk[2])**2)
    
    # order p=1 (monopole)
    a10 = mu0 * (1. + np.conj(Wk[2]))
    a01 = mu0 * 1j * (1. - np.conj(Wk[2]))
    
    # order p=2
    Q20 = Wk[3] * a10 * a10
    Q11 = Wk[3] * a10 * a01
    Q02 = Wk[3] * a01 * a01
    a20 =  _akl(mu0, Wk[2], Q20)
    a11 =  _akl(mu0, Wk[2], Q11)
    a02 =  _akl(mu0, Wk[2], Q02)
    
    # order p=3 (quadrupole)
    Q30 = Wk[3] * 3. * a20 * a10 + Wk[4] * a10 * a10 * a10
    Q21 = Wk[3] * ( 2.*a11*a10 + a01*a20 ) + Wk[4] * a01 * a10 * a10
    Q12 = Wk[3] * ( 2. * a11 * a01 + a10 * a02 ) + Wk[4] * a10 * a01 * a01
    Q03 = Wk[3] * 3. * a02 * a01 + Wk[4] * a01 * a01 * a01
    a30 =  _akl(mu0, Wk[2], Q30)
    a21 =  _akl(mu0, Wk[2], Q21)
    a12 =  _akl(mu0, Wk[2], Q12)
    a03 =  _akl(mu0, Wk[2], Q03)
    
    # order p=4
    Q40 = Wk[3] * ( 4.*a30*a10 + 3.*a20*a20 ) + Wk[4] * 6.*a20*a10*a10 + Wk[5] * a10*a10*a10*a10
    Q31 = Wk[3] * ( 3.*a21*a10 + 3.*a11*a20 + a01*a30 ) + Wk[4] * 3.*( a11*a10*a10 + a01*a20*a10 ) + Wk[5] * a01*a10*a10*a10
    Q22 = Wk[3] * ( 2.*a12*a10 + 2.*a11*a11 + a02*a20 + 2.*a21*a01 ) + Wk[4] * ( a02*a10*a10 + 4.*a11*a01*a10 + a01*a01*a20 ) + Wk[5] * a01*a01*a10*a10
    Q13 = Wk[3] * ( 3.*a12*a01 + 3.*a11*a02 + a10*a03 ) + Wk[4] * 3.*( a11*a01*a01 + a10*a02*a01 ) + Wk[5] * a10*a01*a01*a01
    Q04 = Wk[3] * ( 4.*a03*a01 + 3.*a02*a02 ) + Wk[4] * 6.*a02*a01*a01 + Wk[5] * a01*a01*a01*a01
    a40 =  _akl(mu0, Wk[2], Q40)
    a31 =  _akl(mu0, Wk[2], Q31)
    a22 =  _akl(mu0, Wk[2], Q22)
    a13 =  _akl(mu0, Wk[2], Q13)
    a04 =  _akl(mu0, Wk[2], Q04)
    
    # order p=5 (hexadecapole)
    Q50 = Wk[3] * ( 5.*a40*a10 + 10.*a20*a30 ) + Wk[4] * ( 10.*a30*a10*a10 + 15.*a20*a20*a10 ) + Wk[5] * 10.*a20*a10*a10*a10 + Wk[6] * a10*a10*a10*a10*a10
    Q41 = Wk[3] * ( 4.*a31*a10 + 4.*a11*a30 + 6.*a21*a20 + a01*a40 ) + Wk[4] * ( 6.*a21*a10*a10 + 12.*a11*a20*a10 + 3.*a01*a20*a20 + 4.*a01*a10*a30 ) + Wk[5] * ( 4.*a11*a10*a10*a10 + 6.*a01*a10*a10*a20 ) + Wk[6] * a01*a10*a10*a10*a10
    Q32 = Wk[3] * ( 3.*a22*a10 + 6.*a11*a21 + 3.*a12*a20 + a02*a30 + 2.*a31*a01 ) + Wk[4] * ( 3.*a12*a10*a10 + 6.*a11*a11*a10 + 3.*a02*a20*a10 + 6.*a01*a11*a20 + 6.*a01*a21*a10 + a01*a01*a30 ) + Wk[5] * ( a02*a10*a10*a10 + 6.*a11*a01*a10*a10 + 3.*a01*a01*a10*a20 ) + Wk[6] * a01*a01*a10*a10*a10
    Q23 = Wk[3] * ( 3.*a22*a01 + 6.*a11*a12 + 3.*a21*a02 + a20*a03 + 2.*a13*a10 ) + Wk[4] * ( 3.*a21*a01*a01 + 6.*a11*a11*a01 + 3.*a20*a02*a01 + 6.*a10*a11*a02 + 6.*a10*a12*a01 + a10*a10*a03 ) + Wk[5] * ( a20*a01*a01*a01 + 6.*a11*a10*a01*a01 + 3.*a10*a10*a01*a02 ) + Wk[6] * a10*a10*a01*a01*a01
    Q14 = Wk[3] * ( 4.*a13*a01+ 4.*a11*a03 + 6.*a12*a02 + a10*a04 ) + Wk[4] * ( 6.*a12*a01*a01 + 12.*a11*a02*a01 + 3.*a10*a02*a02 + 4.*a10*a01*a03 ) + Wk[5] * ( 4.*a11*a01*a01*a01 + 6.*a10*a01*a01*a02 ) + Wk[6] * a10*a01*a01*a01*a01
    Q05 = Wk[3] * ( 5.*a04*a01 + 10.*a02*a03 ) + Wk[4] * ( 10.*a03*a01*a01 + 15.*a02*a02*a01 ) + Wk[5] * 10.*a02*a01*a01*a01 + Wk[6] * a01*a01*a01*a01*a01
    a50 =  _akl(mu0, Wk[2], Q50)
    a41 =  _akl(mu0, Wk[2], Q41)
    a32 =  _akl(mu0, Wk[2], Q32)
    a23 =  _akl(mu0, Wk[2], Q23)
    a14 =  _akl(mu0, Wk[2], Q14)
    a05 =  _akl(mu0, Wk[2], Q05)
    
    # compute hexadecapole and quadrupolar mu and A
    mu2 = 1./4. * np.imag(a01 * np.conj(a12 + a30) + np.conj(a10) * (a03 + a21) + 2. * a02 * np.conj(a11) + 2. * a11 * np.conj(a20))
    mu4 = 1./8. * np.imag((a05 + 2. * a23 + a41) * np.conj(a10) + 4. * a04 * np.conj(a11) + 4 * (a13 + a31) * np.conj(a20) + 6. * a12 * np.conj(a21) + 6. * a21 * np.conj(a30) + a03 * (6. * np.conj(a12) + 2. * np.conj(a30)) + 4. * a02 * np.conj(a13 + a31) + 4. * a11 * np.conj(a40) + a01 * np.conj(a14 + 2. * a32 + a50))
    A0 = np.sum(np.abs(mu0))
    A2 = np.sum(np.abs(mu0 + 1. / 2. * mu2 * (1. - 1./5. * Gamma) * rho**2))
    A4 = np.sum(np.abs(mu0 + 1. / 2. * mu2 * (1. - 1./5. * Gamma) * rho**2 + 1. / 24. * mu4 * (1. - 11. / 35. * Gamma) * rho**4))
    
    return np.array([A0, A2, A4])

def _akl(mu, W2, Qkl):
    """Compute a(p-n, n) factors from Q(p-n, n)"""
    akl = mu*(np.conj(Qkl) + np.conj(W2) * Qkl)
    return akl

def _solvelenseq(s, q, zeta):
    """Solve binary lens equation [convention Cassan (2008)]"""
    coefs = [(1. + q)**2 * (s + np.conj(zeta)) * np.conj(zeta),
             (1. + q) * (s * (q - np.abs(zeta)**2 * (1. + q)) + (1. + q) * ((1. + 2 * s**2) - np.abs(zeta)**2 + 2. * s * np.conj(zeta)) * zeta.conjugate()),
             (1. + q) * (s**2 * q - s * (1. + q) * zeta + (2 * s + s**3 * (1. + q) + s**2 * (1. + q) * np.conj(zeta)) * np.conj(zeta) - 2. * np.abs(zeta)**2 * (1. + q) * (1. + s**2 + s * np.conj(zeta))),
             - (1. + q) * (s * q + s**2 * (q - 1.) * np.conj(zeta) + (1. + q + s**2 * (2. + q)) * zeta + np.abs(zeta)**2 * (2. * s * (2. + q) + s**2 * (1. + q) * (s + np.conj(zeta)))),
             - s * (1. + q) * ((2. + s**2) * zeta + 2. * s * np.abs(zeta)**2) - s**2 * q,
             - s**2 * zeta]
    return np.roots(coefs)

def example():
    """Example"""
    # binary lens and source parameters
    s = 1.7
    q = 0.2
    rho = 0.01
    Gamma = 0.
    
    # center of source (ref. center of mass)
    zeta0 = np.complex(-.5,0.)
    zeta0 = zeta0 - np.complex(s * q / (1. + q), 0.)
    
    # solve lens equation for the source center
    z0 = _solvelenseq(s, q, zeta0)
    W1 = 1. / (1. + q) * (1. / z0 + q / (z0 + s))
    z0 = z0[np.abs(z0 - W1.conjugate() - zeta0) < 0.000001]
    # ... to be computed for all source center positions:
    nr = len(z0)
    Wk = np.empty((7, nr), dtype=np.complex128)
    
    # monopole (A0) + quadrupole (A2)
    Wk[2] = -1. / (1. + q) * (1. / z0**2 + q / (z0 + s)**2)
    Wk[3] = 2. / (1. + q) * (1. / z0**3 + q / (z0 + s)**3)
    Wk[4] = -6. / (1. + q) * (1. / z0**4 + q / (z0 + s)**4)
    A0, A2 = quadrupole(Wk, rho, Gamma)
    print(" A0, A2 = ", A0, A2)
    print(" ... should give: 6.08605 6.17629")
    
    # monopole (A0) + quadrupole (A2) + hexadecapole (A4)
    Wk[2] = -1. / (1. + q) * (1. / z0**2 + q / (z0 + s)**2)
    Wk[3] = 2. / (1. + q) * (1. / z0**3 + q / (z0 + s)**3)
    Wk[4] = -6. / (1. + q)*(1. / z0**4 + q / (z0 + s)**4)
    Wk[5] = 24. / (1. + q) * (1. / z0**5 + q / (z0 + s)**5)
    Wk[6] = -120. / (1. + q) * (1. / z0**6 + q / (z0 + s)**6)
    A0, A2, A4 = hexadecapole(Wk, rho, Gamma)
    print(" A0, A2, A4 = ", A0, A2, A4)
    print(" ... should give: 6.08605 6.17629 6.18172")

if __name__ == "__main__":
    example()
