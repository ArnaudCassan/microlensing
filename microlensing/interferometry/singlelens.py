# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

import numpy as np
from scipy.integrate import romberg

def V_unif_exact(u1, rho, uE, vE, tol=1e-4, sing=1e-5, **kwargs):
    """Complex visibility: exact formula.
    [version: 5.9.21]
    """
    
    # check parameters
    if rho <= 0.: raise ValueError("rho must be strictly positive")
    if u1 < 0.: raise ValueError("u1 must be positive")

    # integration boundary betam
    if u1 == 0.:
    # Perfect Einstein ring
        betam = np.pi / 2.
    else:
        betam = np.arcsin(np.amin([rho / u1, 1.]))
    
    # scale factor for accuracy tolerance in integration
    K = np.pi * rho**2

    # function to integrate, real part
    def integ_Re(beta, u1, rho, uE, vE):
        
        A = u1 * np.cos(beta)
        B = np.sqrt(np.abs(rho**2 - (u1 * np.sin(beta))**2))
        up = A + B
        um = A - B
        
        rpp = 0.5 * (up + np.sqrt(up**2 + 4.))
        rpm = 0.5 * (um + np.sqrt(um**2 + 4.))
        
        rmp = 0.5 * (up - np.sqrt(up**2 + 4.))
        rmm = 0.5 * (um - np.sqrt(um**2 + 4.))
        
        Omega = - uE * np.sin(beta) + vE * np.cos(beta)
        
        if np.abs(Omega) <= sing:
            return 0.5 * (rpp**2 - rpm**2 - rmp**2 + rmm**2) / K
        else:
            return ((np.cos(2. * np.pi * Omega * rpp) - np.cos(2. * np.pi * Omega * rpm) - np.cos(2. * np.pi * Omega * rmp) + np.cos(2. * np.pi * Omega * rmm)) / (4. * Omega**2 * np.pi**2) + (np.sin(2. * np.pi * Omega * rpp) * rpp - np.sin(2. * np.pi * Omega * rpm) * rpm - np.sin(2. * np.pi * Omega * rmp) * rmp + np.sin(2. * np.pi * Omega * rmm) * rmm) / (2. * Omega * np.pi)) / K
    
    # function to integrate, imaginary part
    def integ_Im(beta, u1, rho, uE, vE):
        
        A = u1 * np.cos(beta)
        B = np.sqrt(np.abs(rho**2 - (u1 * np.sin(beta))**2))
        up = A + B
        um = A - B
        
        rpp = 0.5 * (up + np.sqrt(up**2 + 4.))
        rpm = 0.5 * (um + np.sqrt(um**2 + 4.))
        
        rmp = 0.5 * (up - np.sqrt(up**2 + 4.))
        rmm = 0.5 * (um - np.sqrt(um**2 + 4.))
        
        Omega = - uE * np.sin(beta) + vE * np.cos(beta)
        
        if np.abs(Omega) <= sing:
            return 0.
        else:
            return ((-np.sin(2. * np.pi * Omega * rpp) + np.sin(2. * np.pi * Omega * rpm) + np.sin(2. * np.pi * Omega * rmp) - np.sin(2. * np.pi * Omega * rmm)) / (4. * Omega**2 * np.pi**2) + (np.cos(2. * np.pi * Omega * rpp) * rpp - np.cos(2. * np.pi * Omega * rpm) * rpm - np.cos(2. * np.pi * Omega * rmp) * rmp + np.cos(2. * np.pi * Omega * rmm) * rmm) / (2. * Omega * np.pi)) / K
 
    # function to integrate, Einstein ring
    def integ_Ein(beta, rho, uE, vE):
        
        C = np.sqrt(rho**2 + 4.)
        rpp = 0.5 * (rho + C)
        rmp = 0.5 * (rho - C)
                
        Omega = - uE * np.sin(beta) + vE * np.cos(beta)
        
        if np.abs(Omega) <= sing:
            return (rpp**2 - rmp**2) / K
        else:
            return 2. * ((np.cos(2. * np.pi * Omega * rpp) - np.cos(2. * np.pi * Omega * rmp)) / (4. * Omega**2 * np.pi**2) + (np.sin(2. * np.pi * Omega * rpp) * rpp - np.sin(2. * np.pi * Omega * rmp) * rmp) / (2. * Omega * np.pi)) / K
 
    if u1 == 0.:
    # Perfect Einstein ring
        # avoid romberg to get stuck when |uE|=|vE|, 10 tries
        for i in range(10):
            b1 = integ_Ein(-betam, rho, uE, vE)
            b2 = integ_Ein(0., rho, uE, vE)
            b3 = integ_Ein(betam, rho, uE, vE)
            if np.abs(b1-b2) + np.abs(b3-b2) < tol:
                uE = uE + 1e-4 * (np.random.rand() - 0.5)
                vE = vE + 1e-4 * (np.random.rand() - 0.5)
            else:
                break
        # compute visibility
        V_Ein = K * romberg(integ_Ein, -betam, betam, args=(rho, uE, vE), tol=tol, **kwargs)
        V0 = K * romberg(integ_Ein, -betam, betam, args=(rho, 0., 0.), tol=tol, **kwargs)
        return V_Ein / V0
    
    else:
        # compute visibility
        V_Re = K * romberg(integ_Re, -betam, betam, args=(u1, rho, uE, vE), tol=tol, **kwargs)
        V_Im = K * romberg(integ_Im, -betam, betam, args=(u1, rho, uE, vE), tol=tol, **kwargs)
        V0 = K * romberg(integ_Re, -betam, betam, args=(u1, rho, 0., 0.), tol=tol, **kwargs)
        return np.complex(V_Re, V_Im) / V0

def V_unif_thinarcs(u1, rho, uE, vE, tol=1e-4, **kwargs):
    """Complex visibility: thin-arcs formula.
    [version: 9.9.21]
    """
    
    # check parameters
    if rho <= 0.: raise ValueError("rho must be stricktly positive")
    if u1 < 0.: raise ValueError("u1 must be positive")

    # integration boundary betam, define eta1, scale
    # factor for accuracy tolerance in integration
    if u1 == 0.:
        # Perfect Einstein ring
        betam = np.pi / 2.
        K = np.pi * rho / 4.
    else:
        betam = np.arcsin(np.amin([rho / u1, 1.]))
        eta1 = rho / u1
        K = np.pi * rho**2 / (4. * u1)
    
    # function to integrate (u1≠0)
    def integ(beta, eta1, uE, vE):
        return (np.sqrt(np.abs(eta1**2 - np.sin(beta)**2)) * np.cos(2. * np.pi * vE * np.cos(beta)) * np.cos(2. * np.pi * uE * np.sin(beta))) / K
        
    # function to integrate (u1=0)
    def integring(beta, uE, vE):
        return (np.cos(2. * np.pi * vE * np.cos(beta)) * np.cos(2. * np.pi * uE * np.sin(beta))) / K
      
    # visibility
    if u1 == 0.:
        # Perfect Einstein ring
        V = K * romberg(integring, 0., betam, args=(uE, vE), tol=tol, **kwargs)
        V0 = K * romberg(integring, 0., betam, args=(0., 0.), tol=tol, **kwargs)
    else:
        V = K * romberg(integ, 0., betam, args=(eta1, uE, vE), tol=tol, **kwargs)
        V0 = K * romberg(integ, 0., betam, args=(eta1, 0., 0.), tol=tol, **kwargs)

    return V / V0
  
def V_pointsource(u1, uE, vE):
    """Complex visibility: point-source formula.
    [version: 5.9.21]
    """
    
    # check parameter
    if u1 <= 0.: raise ValueError("u1 must be stricktly positive")
    
    # images positions
    yp = 0.5 * (u1 + np.sqrt(u1**2 + 4.))
    ym = 0.5 * (u1 - np.sqrt(u1**2 + 4.))
    
    # ps magnification
    B = (u1**2 + 2.) / (u1 * np.sqrt(u1**2 + 4.))
    mup = np.abs(0.5 * (1. + B))
    mum = np.abs(0.5 * (1. - B))

    # visibility
    Fp = mup * np.exp(np.complex(0., -2. * np.pi * vE * yp))
    Fm = mum * np.exp(np.complex(0., -2. * np.pi * vE * ym))
    
    V = Fp + Fm
    V0 = mup + mum
    
    return V / V0


if __name__ == '__main__':

    rho = 0.03
    u1 = 0.04
    uE = 0.2
    vE = 0.18
    
    Vex = V_unif_exact(0., rho, uE, vE, tol=1e-4)
    Vth = V_unif_thinarcs(0., rho, uE, vE, tol=1e-4)
    
    print(f"  uE: {uE}, vE: {vE}")
    print(f"  V^2 Exact.......  {Vex}")
    print(f"  V^2 Thin arcs...  {Vth}")
    print(f"  ∆V^2............  {Vth-Vex}")
    
