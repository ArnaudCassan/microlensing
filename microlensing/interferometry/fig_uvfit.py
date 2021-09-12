# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 19/11/2020

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from astropy.time import Time
from utils import verbosity, printi, printd, printw
from ESPL import mod_VIS
from data import obs_VIS2
from get_bestfit import get_bestfit

def fig_uvfit(oifits, take, model, alpha, rho, thetaE, u0, tE, t0, LLD=0., Na=8, Nli=0, sigresc=(1., 0.), figname=None, ax=None):
    """Plot PIONIER fit and measurements per epoch and per take.
    
    Warning
    -------
    Convention (microlensing): North (N) is up, East (E) is left.
    
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[30m", "\033[0m", "\033[0m\033[3m"

    # infos
    printi(tcol + "Using model : " + tun + model + tend)

    # reference date (MJD)
    ref_MJD = 58681.
    
    # maximum excursion in module of B/l
    maxmoduv = 0.4

    # get date of take
    uvdata = obs_VIS2(oifits, 1, 1, ref_MJD=ref_MJD)
    DATE_ref = uvdata[take][4]

    # create rotated trajectory
    zetac = np.complex((DATE_ref - t0) / tE, u0) * np.exp(np.complex(0., np.pi / 180. * alpha))

    # rainbow colors
    Bcols = [cm.gist_rainbow(a) for a in np.linspace(0., 1., 6)]
    Bcols.reverse()
    
    # plot design
    if ax is None:
        plt.close('all')
        plt.subplots(figsize=(2.5, 3))
        plt.subplots_adjust(top = 0.96, bottom=0.16, left=0.23, right=0.95)
        ax = plt.subplot(1, 1, 1)

    # plot size and labels
    #    ax.set_title(r'Einstein squared visibility')
    ax.set_xlabel(r'$B/\lambda$ $[\theta_{\rm E}^{-1}]$')
    ax.set_ylabel(r'$|V_{\rm E}|^2$')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, maxmoduv])
    
    # conversion to mas
    mas = np.pi / 180. / 3600. / 1000.
            
    # compute and plot observed V^2(u,v)
    VIS2RES = []
    tol=1e-5
    for b in tqdm(range(6)):
        lmoduv, lVISE2 = [], []
        for l in range(6):
            Bcol = Bcols[l]
            u, v, VIS2, VIS2ERR, MJD = obs_VIS2(oifits, b + 1, l + 1, ref_MJD=ref_MJD, sigresc=sigresc)[take]
            u = u * mas * thetaE
            v = v * mas * thetaE
            moduv = (u**2 + v**2)**0.5
            VISE2 = np.abs(mod_VIS(zetac, rho, u, v, model, LLD=LLD, Na=Na, tol=tol))**2
            lmoduv.append(moduv)
            lVISE2.append(VISE2)
                        
            ax.scatter(moduv, VISE2, c='black', s=0.7, marker='o', lw=0.9)
            ax.errorbar(moduv, VIS2, VIS2ERR, c=Bcol, lw=0.8, zorder=20, fmt='o', ms=2)
            
            # store VIS^2 residuals and error bars
            VIS2RES.append([b, l, VIS2 - VISE2, VIS2ERR])
    
        # plot model (line)
        ax.plot(np.array(lmoduv), np.array(lVISE2), color='black', lw=1)
        
        # full model lines
        lmoduv, lVISE2 = [], []
        # take the last value of (u, v)
        ub, vb = u, v
        u = ub / (ub**2 + vb**2)**0.5 * np.linspace(0., maxmoduv, Nli)
        v = vb / (ub**2 + vb**2)**0.5 * np.linspace(0., maxmoduv, Nli)
        for i in range(Nli):
            moduv = (u[i]**2 + v[i]**2)**0.5
            lmoduv.append(moduv)
            lVISE2.append(np.abs(mod_VIS(zetac, rho, u[i], v[i], model, LLD=LLD, Na=Na, tol=tol))**2)
        ax.plot(np.array(lmoduv), np.array(lVISE2), color='gray', lw=0.3)
    
    #    # save VIS^2 residuals and error bars
    #    VIS2RES = np.array(VIS2RES)
    #    np.savetxt(figname + '_residuals.out', VIS2RES)

    # add date+time labels (date + UTC time)
    t_JD = 2400000.5 + DATE_ref + ref_MJD
    t = Time(t_JD, format='jd')
    ax.text(0.365, 0.56, r'{}'.format(t.iso[0:16]), fontsize=8, rotation=90)
    
    # print dates in JD-2458681
    printi(tit + "Data set: {0}, obs. {1} taken at (JD-2458681) t=".format(oifits, take+1) + tend + str(t_JD-2458681))
    
    if figname is not None:
        plt.savefig(figname)

    
if __name__ == "__main__":
        
    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # choose visibility model
    #model = 'ESPL_PS_MULTIEP'
    model = 'ESPL_FLAT_MULTIEP'
    
    # best-fit parameters
    alpha, rho, thetaE, u0, tE, t0 = get_bestfit('mcmc_all_series_11.h5')
        
    # convert to East-right convention of the code
    alpha = 180. - alpha
    u0 = -u0
    
    # linear limb-darkening 'a' parameter
    LLD = 0.45
    Na = 8
    
    # error bars (rescale, systematics)
    sigresc = (1., 0.03)
    
    # sampling for model lines
    Nli = 0 # 20
    
    # July 12, take 1, take 2
    oifits = '2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits'

    fig_uvfit(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresc, figname='uvfit-12A.pdf')

    fig_uvfit(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresc, figname='uvfit-12B.pdf')

    # July 19, take 1, take 2
    oifits = '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits'

    fig_uvfit(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresc, figname='uvfit-19A.pdf')

    fig_uvfit(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresc, figname='uvfit-19B.pdf')

    # July 21, take 1
    oifits = '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits'

    fig_uvfit(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresc, figname='uvfit-21A.pdf')

