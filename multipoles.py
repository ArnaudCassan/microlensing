# -*- coding: utf-8 -*-

#   Original version: 2017, Arnaud Cassan
#
#

import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def example():
    """Example of using quadrupole(Wk,rho,Gamma) and hexadecapole(Wk,rho,Gamma)."""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Binary lens and source parameters
    s = 1.7
    q = 0.2
    rho = 0.01
    Gamma = 0.
### Center of source (ref. center of mass)
    zeta0 = np.complex(-.5,0.)
    zeta0 = zeta0 - np.complex(s*q/(1.+q),0.)
### Solve lens equation for the center
    z0 = solve_lens_poly(s,q,zeta0)
    W1 = 1./(1.+q)*(1./z0+q/(z0+s))
    z0 = z0[np.abs(z0-W1.conjugate()-zeta0)<0.000001]
### To be computed for all source center positions:
    nr = len(z0)
    Wk = np.empty((7,nr),dtype=np.complex128)
# Choice 1: Quadrupole only (A2)
    Wk[2] = -1./(1.+q)*(1./z0**2+q/(z0+s)**2)
    Wk[3] = 2./(1.+q)*(1./z0**3+q/(z0+s)**3)
    Wk[4] = -6./(1.+q)*(1./z0**4+q/(z0+s)**4)
    A0,A2 = quadrupole(Wk,rho,Gamma)
# Choice 2: Quadrupole (A2) and Hexadecapole (A4)
    Wk[2] = -1./(1.+q)*(1./z0**2+q/(z0+s)**2)
    Wk[3] = 2./(1.+q)*(1./z0**3+q/(z0+s)**3)
    Wk[4] = -6./(1.+q)*(1./z0**4+q/(z0+s)**4)
    Wk[5] = 24./(1.+q)*(1./z0**5+q/(z0+s)**5)
    Wk[6] = -120./(1.+q)*(1./z0**6+q/(z0+s)**6)
    A0,A2,A4 = hexadecapole(Wk,rho,Gamma)
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def quadrupole(Wk,rho,Gamma):
    """Quadrupole expansion of the finite-source magnification."""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Monopole
    mu0 = 1./(1.-np.abs(Wk[2])**2)
## Order p=1 (monopole)
    a10 = mu0*(1.+Wk[2].conjugate())
    a01 = mu0*(1j)*(1.-Wk[2].conjugate())
## Order p=2
    Q20 = Wk[3] * ( a10 * a10 )
    Q11 = Wk[3] * ( a10 * a01 )
    Q02 = Wk[3] * ( a01 * a01 )
    a20 =  akl(mu0,Wk[2],Q20)
    a11 =  akl(mu0,Wk[2],Q11)
    a02 =  akl(mu0,Wk[2],Q02)
## Order p=3 (quadrupole)
    Q30 = Wk[3] * ( a20 * a10 + a20 * a10 + a10 * a20 ) + Wk[4] * ( a10 * a10 * a10 )
    Q21 = Wk[3] * ( a11 * a10 + a11 * a10 + a01 * a20 ) + Wk[4] * ( a01 * a10 * a10 )
    Q12 = Wk[3] * ( a11 * a01 + a02 * a10 + a01 * a11 ) + Wk[4] * ( a01 * a10 * a01 )
    Q03 = Wk[3] * ( a02 * a01 + a02 * a01 + a01 * a02 ) + Wk[4] * ( a01 * a01 * a01 )
    a30 =  akl(mu0,Wk[2],Q30)
    a21 =  akl(mu0,Wk[2],Q21)
    a12 =  akl(mu0,Wk[2],Q12)
    a03 =  akl(mu0,Wk[2],Q03)
### Quadrupole mu and A
    mu2 = 1./4.*np.imag( a01*(a12+a30).conjugate() + a10.conjugate()*(a03+a21) + 2.*a02*a11.conjugate() + 2.*a11*a20.conjugate() )
    A0 = np.sum(np.abs(mu0))
    A2 = np.sum(np.abs(mu0+1/2.*mu2*(1.-1./5.*Gamma)*rho**2))
    return A0, A2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def hexadecapole(Wk,rho,Gamma):
    """Quadrupole and Hexadecapole expansions of the finite-source magnification"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Monopole
    mu0 = 1./(1.-np.abs(Wk[2])**2)
## Order p=1 (monopole)
    a10 = mu0*(1.+Wk[2].conjugate())
    a01 = mu0*(1j)*(1.-Wk[2].conjugate())
    ## Order p=2
    Q20 = Wk[3] * ( a10 * a10 )
    Q11 = Wk[3] * ( a10 * a01 )
    Q02 = Wk[3] * ( a01 * a01 )
    a20 =  akl(mu0,Wk[2],Q20)
    a11 =  akl(mu0,Wk[2],Q11)
    a02 =  akl(mu0,Wk[2],Q02)
## Order p=3 (quadrupole)
    Q30 = Wk[3] * ( a20 * a10 + a20 * a10 + a10 * a20 ) + Wk[4] * ( a10 * a10 * a10 )
    Q21 = Wk[3] * ( a11 * a10 + a11 * a10 + a01 * a20 ) + Wk[4] * ( a01 * a10 * a10 )
    Q12 = Wk[3] * ( a11 * a01 + a02 * a10 + a01 * a11 ) + Wk[4] * ( a01 * a10 * a01 )
    Q03 = Wk[3] * ( a02 * a01 + a02 * a01 + a01 * a02 ) + Wk[4] * ( a01 * a01 * a01 )
    a30 =  akl(mu0,Wk[2],Q30)
    a21 =  akl(mu0,Wk[2],Q21)
    a12 =  akl(mu0,Wk[2],Q12)
    a03 =  akl(mu0,Wk[2],Q03)
## Order p=4
    Q40 = Wk[3] * ( a30 * a10 + a20 * a20 + a30 * a10 + a20 * a20 + a20 * a20 + a30 * a10 + a10 * a30 ) + Wk[4] * ( a20 * a10 * a10 + a20 * a10 * a10 + a20 * a10 * a10 + a10 * a20 * a10 + a10 * a20 * a10 + a10 * a10 * a20 ) + Wk[5] * ( a10 * a10 * a10 * a10 )
    Q31 = Wk[3] * ( a21 * a10 + a11 * a20 + a21 * a10 + a11 * a20 + a11 * a20 + a21 * a10 + a01 * a30 ) + Wk[4] * ( a11 * a10 * a10 + a11 * a10 * a10 + a11 * a10 * a10 + a01 * a20 * a10 + a01 * a20 * a10 + a01 * a10 * a20 ) + Wk[5] * ( a01 * a10 * a10 * a10 )
    Q22 = Wk[3] * ( a12 * a10 + a11 * a11 + a12 * a10 + a11 * a11 + a02 * a20 + a21 * a01 + a01 * a21 ) + Wk[4] * ( a02 * a10 * a10 + a11 * a01 * a10 + a11 * a01 * a10 + a01 * a11 * a10 + a01 * a11 * a10 + a01 * a01 * a20 ) + Wk[5] * ( a01 * a01 * a10 * a10 )
    Q13 = Wk[3] * ( a12 * a01 + a02 * a11 + a03 * a10 + a11 * a02 + a02 * a11 + a12 * a01 + a01 * a12 ) + Wk[4] * ( a02 * a10 * a01 + a11 * a01 * a01 + a02 * a01 * a10 + a01 * a11 * a01 + a01 * a02 * a10 + a01 * a01 * a11 ) + Wk[5] * ( a01 * a01 * a10 * a01 )
    Q04 = Wk[3] * ( a03 * a01 + a02 * a02 + a03 * a01 + a02 * a02 + a02 * a02 + a03 * a01 + a01 * a03 ) + Wk[4] * ( a02 * a01 * a01 + a02 * a01 * a01 + a02 * a01 * a01 + a01 * a02 * a01 + a01 * a02 * a01 + a01 * a01 * a02 ) + Wk[5] * ( a01 * a01 * a01 * a01 )
    a40 =  akl(mu0,Wk[2],Q40)
    a31 =  akl(mu0,Wk[2],Q31)
    a22 =  akl(mu0,Wk[2],Q22)
    a13 =  akl(mu0,Wk[2],Q13)
    a04 =  akl(mu0,Wk[2],Q04)
## Order p=5 (hexadecapole)
    Q50 = Wk[3] * ( a40 * a10 + a20 * a30 + a30 * a20 + a30 * a20 + a40 * a10 + a20 * a30 + a30 * a20 + a30 * a20 + a30 * a20 + a30 * a20 + a40 * a10 + a20 * a30 + a20 * a30 + a40 * a10 + a10 * a40 ) + Wk[4] * ( a30 * a10 * a10 + a20 * a20 * a10 + a20 * a20 * a10 + a30 * a10 * a10 + a20 * a20 * a10 + a20 * a20 * a10 + a30 * a10 * a10 + a20 * a20 * a10 + a20 * a20 * a10 + a20 * a20 * a10 + a30 * a10 * a10 + a20 * a10 * a20 + a20 * a20 * a10 + a30 * a10 * a10 + a20 * a10 * a20 + a20 * a10 * a20 + a20 * a10 * a20 + a30 * a10 * a10 + a10 * a30 * a10 + a10 * a20 * a20 + a10 * a30 * a10 + a10 * a20 * a20 + a10 * a20 * a20 + a10 * a30 * a10 + a10 * a10 * a30 ) + Wk[5] * ( a20 * a10 * a10 * a10 + a20 * a10 * a10 * a10 + a20 * a10 * a10 * a10 + a20 * a10 * a10 * a10 + a10 * a20 * a10 * a10 + a10 * a20 * a10 * a10 + a10 * a20 * a10 * a10 + a10 * a10 * a20 * a10 + a10 * a10 * a20 * a10 + a10 * a10 * a10 * a20 ) + Wk[6] * ( a10 * a10 * a10 * a10 * a10 )
    Q41 = Wk[3] * ( a31 * a10 + a11 * a30 + a21 * a20 + a21 * a20 + a31 * a10 + a11 * a30 + a21 * a20 + a21 * a20 + a21 * a20 + a21 * a20 + a31 * a10 + a11 * a30 + a11 * a30 + a31 * a10 + a01 * a40 ) + Wk[4] * ( a21 * a10 * a10 + a11 * a20 * a10 + a11 * a20 * a10 + a21 * a10 * a10 + a11 * a20 * a10 + a11 * a20 * a10 + a21 * a10 * a10 + a11 * a20 * a10 + a11 * a20 * a10 + a11 * a20 * a10 + a21 * a10 * a10 + a11 * a10 * a20 + a11 * a20 * a10 + a21 * a10 * a10 + a11 * a10 * a20 + a11 * a10 * a20 + a11 * a10 * a20 + a21 * a10 * a10 + a01 * a30 * a10 + a01 * a20 * a20 + a01 * a30 * a10 + a01 * a20 * a20 + a01 * a20 * a20 + a01 * a30 * a10 + a01 * a10 * a30 ) + Wk[5] * ( a11 * a10 * a10 * a10 + a11 * a10 * a10 * a10 + a11 * a10 * a10 * a10 + a11 * a10 * a10 * a10 + a01 * a20 * a10 * a10 + a01 * a20 * a10 * a10 + a01 * a20 * a10 * a10 + a01 * a10 * a20 * a10 + a01 * a10 * a20 * a10 + a01 * a10 * a10 * a20 ) + Wk[6] * ( a01 * a10 * a10 * a10 * a10 )
    Q32 = Wk[3] * ( a22 * a10 + a11 * a21 + a12 * a20 + a21 * a11 + a22 * a10 + a11 * a21 + a12 * a20 + a21 * a11 + a12 * a20 + a21 * a11 + a22 * a10 + a11 * a21 + a02 * a30 + a31 * a01 + a01 * a31 ) + Wk[4] * ( a12 * a10 * a10 + a11 * a11 * a10 + a11 * a11 * a10 + a12 * a10 * a10 + a11 * a11 * a10 + a11 * a11 * a10 + a12 * a10 * a10 + a11 * a11 * a10 + a11 * a11 * a10 + a02 * a20 * a10 + a21 * a01 * a10 + a11 * a01 * a20 + a02 * a20 * a10 + a21 * a01 * a10 + a11 * a01 * a20 + a02 * a10 * a20 + a11 * a01 * a20 + a21 * a01 * a10 + a01 * a21 * a10 + a01 * a11 * a20 + a01 * a21 * a10 + a01 * a11 * a20 + a01 * a11 * a20 + a01 * a21 * a10 + a01 * a01 * a30 ) + Wk[5] * ( a02 * a10 * a10 * a10 + a11 * a01 * a10 * a10 + a11 * a01 * a10 * a10 + a11 * a01 * a10 * a10 + a01 * a11 * a10 * a10 + a01 * a11 * a10 * a10 + a01 * a11 * a10 * a10 + a01 * a01 * a20 * a10 + a01 * a01 * a20 * a10 + a01 * a01 * a10 * a20 ) + Wk[6] * ( a01 * a01 * a10 * a10 * a10 )
    Q23 = Wk[3] * ( a13 * a10 + a11 * a12 + a12 * a11 + a12 * a11 + a13 * a10 + a11 * a12 + a12 * a11 + a12 * a11 + a03 * a20 + a21 * a02 + a22 * a01 + a02 * a21 + a02 * a21 + a22 * a01 + a01 * a22 ) + Wk[4] * ( a03 * a10 * a10 + a11 * a02 * a10 + a11 * a02 * a10 + a12 * a01 * a10 + a02 * a11 * a10 + a11 * a11 * a01 + a12 * a01 * a10 + a02 * a11 * a10 + a11 * a11 * a01 + a02 * a11 * a10 + a12 * a01 * a10 + a11 * a01 * a11 + a02 * a11 * a10 + a12 * a01 * a10 + a11 * a01 * a11 + a02 * a01 * a20 + a02 * a01 * a20 + a21 * a01 * a01 + a01 * a12 * a10 + a01 * a11 * a11 + a01 * a12 * a10 + a01 * a11 * a11 + a01 * a02 * a20 + a01 * a21 * a01 + a01 * a01 * a21 ) + Wk[5] * ( a02 * a01 * a10 * a10 + a02 * a01 * a10 * a10 + a11 * a01 * a01 * a10 + a11 * a01 * a01 * a10 + a01 * a02 * a10 * a10 + a01 * a11 * a01 * a10 + a01 * a11 * a01 * a10 + a01 * a01 * a11 * a10 + a01 * a01 * a11 * a10 + a01 * a01 * a01 * a20 ) + Wk[6] * ( a01 * a01 * a01 * a10 * a10 )
    Q14 = Wk[3] * ( a13 * a01 + a02 * a12 + a03 * a11 + a12 * a02 + a04 * a10 + a11 * a03 + a12 * a02 + a03 * a11 + a03 * a11 + a12 * a02 + a13 * a01 + a02 * a12 + a02 * a12 + a13 * a01 + a01 * a13 ) + Wk[4] * ( a03 * a10 * a01 + a11 * a02 * a01 + a02 * a02 * a10 + a12 * a01 * a01 + a02 * a11 * a01 + a02 * a11 * a01 + a03 * a01 * a10 + a02 * a02 * a10 + a11 * a02 * a01 + a02 * a11 * a01 + a12 * a01 * a01 + a02 * a01 * a11 + a02 * a02 * a10 + a03 * a01 * a10 + a11 * a01 * a02 + a02 * a01 * a11 + a02 * a01 * a11 + a12 * a01 * a01 + a01 * a12 * a01 + a01 * a02 * a11 + a01 * a03 * a10 + a01 * a11 * a02 + a01 * a02 * a11 + a01 * a12 * a01 + a01 * a01 * a12 ) + Wk[5] * ( a02 * a01 * a10 * a01 + a02 * a01 * a10 * a01 + a11 * a01 * a01 * a01 + a02 * a01 * a01 * a10 + a01 * a02 * a10 * a01 + a01 * a11 * a01 * a01 + a01 * a02 * a01 * a10 + a01 * a01 * a11 * a01 + a01 * a01 * a02 * a10 + a01 * a01 * a01 * a11 ) + Wk[6] * ( a01 * a01 * a01 * a10 * a01 )
    Q05 = Wk[3] * ( a04 * a01 + a02 * a03 + a03 * a02 + a03 * a02 + a04 * a01 + a02 * a03 + a03 * a02 + a03 * a02 + a03 * a02 + a03 * a02 + a04 * a01 + a02 * a03 + a02 * a03 + a04 * a01 + a01 * a04 ) + Wk[4] * ( a03 * a01 * a01 + a02 * a02 * a01 + a02 * a02 * a01 + a03 * a01 * a01 + a02 * a02 * a01 + a02 * a02 * a01 + a03 * a01 * a01 + a02 * a02 * a01 + a02 * a02 * a01 + a02 * a02 * a01 + a03 * a01 * a01 + a02 * a01 * a02 + a02 * a02 * a01 + a03 * a01 * a01 + a02 * a01 * a02 + a02 * a01 * a02 + a02 * a01 * a02 + a03 * a01 * a01 + a01 * a03 * a01 + a01 * a02 * a02 + a01 * a03 * a01 + a01 * a02 * a02 + a01 * a02 * a02 + a01 * a03 * a01 + a01 * a01 * a03 ) + Wk[5] * ( a02 * a01 * a01 * a01 + a02 * a01 * a01 * a01 + a02 * a01 * a01 * a01 + a02 * a01 * a01 * a01 + a01 * a02 * a01 * a01 + a01 * a02 * a01 * a01 + a01 * a02 * a01 * a01 + a01 * a01 * a02 * a01 + a01 * a01 * a02 * a01 + a01 * a01 * a01 * a02 ) + Wk[6] * ( a01 * a01 * a01 * a01 * a01 )
    a50 =  akl(mu0,Wk[2],Q50)
    a41 =  akl(mu0,Wk[2],Q41)
    a32 =  akl(mu0,Wk[2],Q32)
    a23 =  akl(mu0,Wk[2],Q23)
    a14 =  akl(mu0,Wk[2],Q14)
    a05 =  akl(mu0,Wk[2],Q05)
### Quadrupole and hexadecapole mu and A
    mu2 = 1./4.*np.imag( a01*(a12+a30).conjugate() + a10.conjugate()*(a03+a21) + 2.*a02*a11.conjugate() + 2.*a11*a20.conjugate() )
    mu4 = 1./8.*np.imag( (a05+2.*a23+a41)*a10.conjugate() + 4.*a04*a11.conjugate() + 4*(a13+a31)*a20.conjugate() + 6.*a12*a21.conjugate() + 6.*a21*a30.conjugate() + a03*(6.*a12.conjugate()+2.*a30.conjugate()) + 4.*a02*(a13+a31).conjugate() + 4.*a11*a40.conjugate() + a01*(a14+2.*a32+a50).conjugate() )
    A0 = np.sum(np.abs(mu0))
    A2 = np.sum(np.abs(mu0+1/2.*mu2*(1.-1./5.*Gamma)*rho**2))
    A4 = np.sum(np.abs(mu0+1/2.*mu2*(1.-1./5.*Gamma)*rho**2+1/24.*mu4*(1.-11./35.*Gamma)*rho**4))
    return A0, A2, A4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def akl(mu,W2,Qkl):
    '''Compute the a(p-n,n) values from Q(p-n,n).'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    akl = mu*(Qkl.conjugate()+W2.conjugate()*Qkl)
    return akl

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def solve_lens_poly(s,q,zeta):
    """Solve binary lens equation [convention Cassan (2008)]."""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    coefs = [(1+q)**2*(s+zeta.conjugate())*zeta.conjugate(),(1+q)*(s*(q-abs(zeta)**2*(1+q))+(1+q)*((1+2*s**2)-abs(zeta)**2+2*s*zeta.conjugate())*zeta.conjugate()),(1+q)*(s**2*q-s*(1+q)*zeta+(2*s+s**3*(1+q)+s**2*(1+q)*zeta.conjugate())*zeta.conjugate()-2*abs(zeta)**2*(1+q)*(1+s**2+s*zeta.conjugate())),-(1+q)*(s*q+s**2*(q-1)*zeta.conjugate()+(1+q+s**2*(2+q))*zeta+abs(zeta)**2*(2*s*(2+q)+s**2*(1+q)*(s+zeta.conjugate()))),-s*(1+q)*((2+s**2)*zeta+2*s*abs(zeta)**2)-s**2*q,-s**2*zeta]
    return np.roots(coefs)




