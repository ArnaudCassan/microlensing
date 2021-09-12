# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 5/12/2020

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from utils import verbosity, printi, printd, printw
from fig_uv import fig_uv
from fig_uvfit import fig_uvfit
from fig_trajarcs import fig_trajarcs
from get_bestfit import get_bestfit

def fig_combined(model, alpha, rho, thetaE, u0, tE, t0, Nuv, LLD, Na, sigresclist, Nli=0, PS=False,  figname=None):
    """Combined figure: uv, uv_fit, arcs_rotation.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[30m", "\033[0m", "\033[0m\033[3m"
    
    # infos
    printi(tcol + "Using model : " + tun + model + tend)
    
    # combined plot design
    fig = plt.figure(constrained_layout=True, figsize=(11.3, 9))
    widths = [3.3, 2.5, 2.5, 3.6]
    heights = [3, 3, 3]
    spec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=widths, height_ratios=heights)
    
    plt.tight_layout()
          
    # fig ref pos
    xp, yp = -0.18, 0.99
    
    # July 12, take 1+2
    oifits = '2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits'

    ax = fig.add_subplot(spec[0, 0])
    ax.text(xp, yp, 'a', weight='bold', transform=ax.transAxes)
    fig_uv(oifits, [0, 1], model, alpha, rho, thetaE, u0, tE, t0, Nuv, ax=ax)

    ax = fig.add_subplot(spec[0, 1])
    fig_uvfit(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresclist[0], ax=ax)

    ax = fig.add_subplot(spec[0, 2])
    fig_uvfit(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresclist[0], ax=ax)

    ax = fig.add_subplot(spec[0, 3])
    fig_trajarcs(oifits, 0, alpha, rho, thetaE, u0, tE, t0, PS=PS, ax=ax)


    # July 19, take 1+2
    oifits = '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits'

    ax = fig.add_subplot(spec[1, 0])
    ax.text(xp, yp, 'b', weight='bold', transform=ax.transAxes)
    fig_uv(oifits, [0, 1], model, alpha, rho, thetaE, u0, tE, t0, Nuv, ax=ax)

    ax = fig.add_subplot(spec[1, 1])
    fig_uvfit(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresclist[1], ax=ax)

    ax = fig.add_subplot(spec[1, 2])
    fig_uvfit(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresclist[1], ax=ax)

    ax = fig.add_subplot(spec[1, 3])
    fig_trajarcs(oifits, 0, alpha, rho, thetaE, u0, tE, t0, PS=PS, ax=ax)


    # July 21, take 1
    oifits = '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits'

    ax = fig.add_subplot(spec[2, 0])
    ax.text(xp, yp, 'c', weight='bold', transform=ax.transAxes)
    fig_uv(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, Nuv, ax=ax)

    ax = fig.add_subplot(spec[2, 1])
    fig_uvfit(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, LLD=LLD, Na=Na, Nli=Nli, sigresc=sigresclist[2], ax=ax)

    ax = fig.add_subplot(spec[2, 3])
    fig_trajarcs(oifits, 0, alpha, rho, thetaE, u0, tE, t0, PS=PS, ax=ax)

    
    plt.savefig(figname)


if __name__ == "__main__":
    
    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # best-fit parameters
    alpha, rho, thetaE, u0, tE, t0 = get_bestfit('mcmc_all_series_11.h5')

    # convert to East-right convention of the code
    alpha = 180. - alpha
    u0 = -u0

    # linear limb-darkening 'a' parameter
    LLD = 0.45
    Na = 8
    
    # error bars rescaling factor
    sigresclist = [(1., 0.03), (1., 0.03), (1., 0.03)]

    # u,v plane sampling (default = 40)
    Nuv = 40
    
    # arcs : combined figure without lines
    model = 'ESPL_FLAT_MULTIEP'
    fig_combined(model, alpha, rho, thetaE, u0, tE, t0, Nuv, LLD, Na, sigresclist, figname='Cassan_Fig2.eps') # 'fig_combined.pdf')

    # point-source : combined figure (SI) without lines
    model = 'ESPL_PS_MULTIEP'
    fig_combined(model, alpha, rho, thetaE, u0, tE, t0, Nuv, LLD, Na, sigresclist, PS=True,  figname='Cassan_EDFig3.eps') # 'fig_combined_PS.pdf')

    # FOR REFEREE ONLY - arcs : combined figure with lines
    model = 'ESPL_FLAT_MULTIEP'
    fig_combined(model, alpha, rho, thetaE, u0, tE, t0, Nuv, LLD, Na, sigresclist, Nli=20, figname='fig_combined_annex.pdf')
