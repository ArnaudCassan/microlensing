# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 16/01/2021

import os, sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from astropy.time import Time
from ESPL import mod_T3PHI
from data import obs_T3PHI, get_triangles
from utils import verbosity, printi, printd, printw
from get_bestfit import get_bestfit

def fig_t3phi(oifits, take, model, alpha, rho, thetaE, u0, tE, t0, LLD=0., Na=8, tol=1e-5, figname=None, ax=None, modcol='black', Tseries=range(4)):
    """Plot fit PIONIER (u, v) measurements per epoch and per take.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[32m", "\033[0m\033[1m\033[32m", "\033[0m", "\033[0m\033[3m"
    
    printi(tcol+"OIFITS file : "+tend+f"{oifits}")
    
    # reference date (MJD)
    ref_MJD = 58681.
    
    # get date of take
    phidata = obs_T3PHI(oifits, 1, 1, ref_MJD=ref_MJD)
    DATE_ref = phidata[take][7]
    
    # create rotated trajectory
    zetac = np.complex((DATE_ref - t0) / tE, u0) * np.exp(np.complex(0., np.pi / 180. * alpha))
            
    # rainbow colors
    Bcols = [cm.gist_rainbow(a) for a in np.linspace(0., 1., 6)]
    Bcols.reverse()
    
    # plot design
    if ax is None:
        plt.close('all')
        plt.subplots(figsize=(5, 1))
        plt.subplots_adjust(top = 0.92, bottom=0.1, left=0.14, right=0.99)
        ax = plt.subplot(1, 1, 1)
    
    # plot size and labels
    ax.set_ylabel(r'$\phi$ (°)')
    ax.set_ylim([-10, 10])
    plt.xticks([])
    plt.yticks([-10, -5, 0, 5, 10])
    ax.grid(ls='-', lw=0.3, axis='y', which='major')
    
    # conversion to mas
    mas = np.pi / 180. / 3600. / 1000.
    
    # compute and plot observed phi
    k, chi2 = 0, 0.
    for T in tqdm(Tseries):
        k += 1 # espace supplémentaire
        lt3phi = []
        for l in range(6):
            Bcol = Bcols[l]
            phidata = obs_T3PHI(oifits, T + 1, l + 1, ref_MJD=ref_MJD)
            U1, V1, U2, V2, T3AMP, T3PHI, T3PHIERR, MJD = phidata[take]
            u_AB = U1 * mas * thetaE
            v_AB = V1 * mas * thetaE
            u_BC = U2 * mas * thetaE
            v_BC = V2 * mas * thetaE
            t3phi, t3amp = mod_T3PHI(zetac, rho, u_AB, v_AB, u_BC, v_BC, model, LLD=LLD, Na=Na, tol=tol)
            lt3phi.append(t3phi)
            
            ax.errorbar(k, T3PHI, T3PHIERR, c=Bcol, lw=0.8, zorder=20, fmt='o', ms=2)
            ax.scatter(k, t3phi, c=modcol, s=4, marker='o',  lw=0.9, zorder=10)
            k += 1
            
            # compute chi2
            if T3PHIERR > 0.:
                chi2 += np.sum((T3PHI - t3phi)**2 / T3PHIERR**2)

        # plot model (line)
        ax.plot(np.arange(k - 6, k), np.array(lt3phi), color=modcol, lw=0.8)
    
    # add date+time labels (date + UTC time)
    t = Time(2400000.5 + DATE_ref + ref_MJD, format='jd')
    ax.text(0.79, 0.85, r'{}'.format(t.iso[0:16]), fontsize=8, transform=ax.transAxes)

    # chi2
    print(f"  | chi2 : {chi2}")

    if figname is not None:
        plt.savefig(figname)
        
def fig_t3phicombined(model, alpha, rho, thetaE, u0, tE, t0, LLD, Na, tol, figname=None):
    """Combined figure.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # combined plot design
    plt.close('all')
    plt.figure(figsize=(6, 6))
    
    # fig ref pos
    xp, yp = -0.14, 0.99
    
    # colors
    alpha1col = 'black'
    alpha2col = 'darkgray'
    
    # July 12, take 1, take 2
    oifits = '2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits'
    
    ax = plt.subplot(5, 1, 1)
    T = (0, 2, 3, 1) # 123, 124, 134, 234
    ax.text(xp, yp, 'a', weight='bold', transform=ax.transAxes)
    fig_t3phi(oifits, 0, model, alpha+180., rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha2col, Tseries=T)
    fig_t3phi(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha1col, Tseries=T)
    ax.set_xticks([3, 10, 17, 24])
    ax.xaxis.set_ticklabels([])
    
    ax = plt.subplot(5, 1, 2)
    T = (3, 0, 2, 1) # 123, 124, 134, 234
    ax.text(xp, yp, 'b', weight='bold', transform=ax.transAxes)
    fig_t3phi(oifits, 1, model, alpha+180., rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha2col, Tseries=T)
    fig_t3phi(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha1col, Tseries=T)
    ax.set_xticks([3, 10, 17, 24])
    ax.xaxis.set_ticklabels([])

    # July 19, take 1, take 2
    oifits = '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits'
        
    ax = plt.subplot(5, 1, 3)
    T = (3, 1, 2, 0) # 123, 124, 134, 234
    ax.text(xp, yp, 'c', weight='bold', transform=ax.transAxes)
    fig_t3phi(oifits, 0, model, alpha+180., rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha2col, Tseries=T)
    fig_t3phi(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha1col, Tseries=T)
    ax.set_xticks([3, 10, 17, 24])
    ax.xaxis.set_ticklabels([])
    
    ax = plt.subplot(5, 1, 4)
    T = (3, 2, 1, 0) # 123, 124, 134, 234
    ax.text(xp, yp, 'd', weight='bold', transform=ax.transAxes)
    fig_t3phi(oifits, 1, model, alpha+180., rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha2col, Tseries=T)
    fig_t3phi(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha1col, Tseries=T)
    ax.set_xticks([3, 10, 17, 24])
    ax.xaxis.set_ticklabels([])
    
    # July 21, take 1
    oifits = '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits'
        
    ax = plt.subplot(5, 1, 5)
    T = (3, 1, 2, 0) # 123, 124, 134, 234
    ax.text(xp, yp, 'e', weight='bold', transform=ax.transAxes)
    fig_t3phi(oifits, 0, model, alpha+180., rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha2col, Tseries=T)
    fig_t3phi(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, ax=ax, modcol=alpha1col, Tseries=T)
    ax.set_xticks([3, 10, 17, 24])
    ax.xaxis.set_ticklabels(['$T_{123}$', '$T_{124}$', '$T_{134}$', '$T_{234}$'])
    
    plt.tight_layout()

    plt.savefig(figname)
    

if __name__ == "__main__":
    
    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # choose visibility model (full, car pas de phase pour thin arcs)
    model = 'ESPL_FULLINT_SINGEP'
    
    # best-fit parameters
    alpha, rho, thetaE, u0, tE, t0 = get_bestfit('mcmc_all_series_11.h5')
        
    # convert to East-right convention of the code
    alpha = 180. - alpha
    u0 = -u0

    # linear limb-darkening 'a' parameter
    LLD = 0.45
    Na = 8
    
    # calc tolerance
    tol = 1e-5

    # check triangles order
#    oifitslist = ['2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits', '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits', '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits']
#    get_triangles(oifitslist[0])

    # combined plot
    fig_t3phicombined(model, alpha, rho, thetaE, u0, tE, t0, LLD, Na, tol, figname='Cassan_EDFig2.eps') # fig_t3phi.pdf
    
    # July 12, take 1, take 2
#    oifits = '2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits'
#
#    fig_t3phi(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, figname='t3phi-12A.pdf')
#
#    fig_t3phi(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, figname='t3phi-12B.pdf')

    # July 19, take 1, take 2
#    oifits = '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits'
#
#    fig_t3phi(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, figname='t3phi-19A.pdf')
#
#    fig_t3phi(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, figname='t3phi-19B.pdf')

    # July 21, take 1
#    oifits = '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits'
#
#    fig_t3phi(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, tol=tol, figname='t3phi-21A.pdf')

   
