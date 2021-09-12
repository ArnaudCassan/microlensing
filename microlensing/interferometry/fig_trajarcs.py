# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 7.9.21, 5/12/2020

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from tqdm import tqdm
from utils import verbosity, printi, printd, printw
from ESPL import newz
from data import obs_VIS2
from get_bestfit import get_bestfit

def fig_trajarcs(oifits, take, alpha, rho, thetaE, u0, tE, t0, PS=False, figname=None, ax=None):
    """Trace fit PIONIER (u, v) measurements per epoch and per take.
    
    Warning
    -------
    Convention (microlensing): North (N) is up, East (E) is left.
    
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

    # reference date (MJD)
    ref_MJD = 58681.

    # get date of take
    uvdata = obs_VIS2(oifits, 1, 1, ref_MJD=ref_MJD)
    DATE_ref = uvdata[take][4]

    # create rotated trajectory
    zetac = np.complex((DATE_ref - t0) / tE, u0) * np.exp(np.complex(0., np.pi / 180. * alpha))
    angle = 180. - np.rad2deg(np.angle(zetac)) % 360
    printi(tcol + "Rotation angle of images (°) : " + tit + str(angle) + tend)
    
    # plot design
    if ax is None:
        plt.close('all')
        plt.subplots(figsize=(3.2, 3))
        plt.subplots_adjust(top = 0.96, bottom=0.16, left=0.17, right=0.98)
        ax = plt.subplot(1, 1, 1)

    # plot size and labels
    #    ax.set_title(r'Images')
    ax.set_xlabel(r'$x_{\rm E}$ $[\theta_{\rm E}]\quad$ West $\longrightarrow$')
    ax.set_ylabel(r'$y_{\rm E}$ $[\theta_{\rm E}]\quad$ North $\longrightarrow$')
    ax.set_xlim([1.5, -1.5])
    ax.set_ylim([-1.4, 1.4])
    ax.set_xticks([-1, 0., 1])
    ax.set_yticks([-1, 0., 1])
    
    # Einstein ring
    ang = 2. * np.pi * np.linspace(0, 1, 100)
    ax.plot([np.cos(p) for p in ang], [np.sin(p) for p in ang], c='lightgray', ls='-', lw=0.6)
    
    # images
    if PS:
        I = np.complex(0., 1.)
        B = np.sqrt(1. + 4. / np.abs(zetac)**2)
        Ep = 0.5 * zetac * (1. + B)
        Em = 0.5 * zetac * (1. - B)
        ax.scatter([np.real(Em), np.real(Ep)], [np.imag(Em), np.imag(Ep)], c=['blue', 'red'], s=20)
    else:
        nzp, nzm = newz(zetac, rho, 30000, 2000)
        nzp[-1], nzm[-1] = nzp[0], nzm[0]
        nxp, nyp = np.real(nzp), np.imag(nzp)
        nxm, nym = np.real(nzm), np.imag(nzm)
        ax.fill_between(nxm, nym, facecolor='blue')
        ax.fill_between(nxp, nyp, facecolor='red')
    
    # lens position
    ax.scatter([0.], [0.], marker='+', c='black', s=15, lw=0.5, zorder=60)
    
    # source trajectory and direction
    zz = 6. * np.exp(np.complex(0., np.pi / 180. * alpha)) * np.array([-0.4, 0.4])
    ax.plot(np.real(zz + zetac), np.imag(zz + zetac), c='black', lw=0.3)
    ax.arrow(np.real(zetac), np.imag(zetac), 0.5 * np.real(zz[1]) / np.abs(zz[1]), 0.5 * np.imag(zz[1]) / np.abs(zz[1]), length_includes_head=True, head_width=0.07, head_length=0.07, lw=0.3, color='k')
#    ax.arrow(-0.3, 0.2, 0.5 * np.real(zz[1]) / np.abs(zz[1]), 0.5 * np.imag(zz[1]) / np.abs(zz[1]), length_includes_head=True, head_width=0.07, head_length=0.07, lw=0.3, color='k')
    
    # draw line connecting images
    x = np.linspace(-2., 2., 2)
    y1 = np.tan(np.deg2rad(-angle)) * x
    ax.plot(x, y1, c='k', lw=0.5, ls='--')
    
    # plot source
    source = Circle((np.real(zetac), np.imag(zetac)), rho, color='goldenrod', fill=True, zorder=30)
    ax.add_artist(source)
    
    # rotation arrow
    angle = angle - 180.
    col = 'k'
    lw = 0.3
    radius = 1.3
    arc = Arc([0., 0.], 2. * radius, 2. * radius, angle=-angle, theta1=-45, theta2=0, capstyle='round', linestyle='-', lw=lw, color=col)
    ax.add_patch(arc)
    beta = -np.deg2rad(45 + angle)
    Xa, Ya = radius * np.cos(beta), radius * np.sin(beta)
    dXa, dYa = -np.sin(beta), np.cos(beta)
    fact = -0.01
    ax.arrow(Xa, Ya, fact * dXa, fact * dYa, length_includes_head=True, head_width=0.07, head_length=0.07, lw=0.3, color=col)
    
    if figname is not None:
        plt.savefig(figname)
    

if __name__ == "__main__":
       
    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')
       
    # visibility model
    model = 'ESPL_FLAT_SINGEP'
           
    # best-fit parameters
    alpha, rho, thetaE, u0, tE, t0 = get_bestfit('mcmc_all_series_11.h5')
        
    # convert to East-right convention of the code
    alpha = 180. - alpha
    u0 = -u0

    # July 12, take 1+2
    oifits = '2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits'
    fig_trajarcs(oifits, 0, alpha, rho, thetaE, u0, tE, t0, figname='trajarcs-12AB.pdf')

    # July 19, take 1+2
    oifits = '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits'
    fig_trajarcs(oifits, 0, alpha, rho, thetaE, u0, tE, t0, figname='trajarcs-19AB.pdf')

    # July 21, take 1
    oifits = '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits'
    fig_trajarcs(oifits, 0, alpha, rho, thetaE, u0, tE, t0, figname='trajarcs-21A.pdf')


