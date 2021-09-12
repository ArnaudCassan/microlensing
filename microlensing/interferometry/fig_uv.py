# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 5/12/2020

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from utils import verbosity, printi, printd, printw
from ESPL import mod_VIS
from data import obs_VIS2
from get_bestfit import get_bestfit

def fig_uv(oifits, take, model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname=None, ax=None):
    """Trace PIONIER (u, v) measurements on top of visibility pattern.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[30m", "\033[0m", "\033[0m\033[3m"
    
    # infos
    printi(tcol + "Using model : " + tun + model + tend)

    # reference date (MJD)
    ref_MJD = 58681.
    
    # combine or not takes into epochs
    if type(take) == list:
        takeref = take[0]
    else:
        takeref = take

    # get date of takeref
    uvdata = obs_VIS2(oifits, 1, 1, ref_MJD=ref_MJD)
    DATE_ref = uvdata[takeref][4]

    # create rotated trajectory
    zetac = np.complex((DATE_ref - t0) / tE, u0) * np.exp(np.complex(0., np.pi / 180. * alpha))

    # theoretical visibility V^2(u,v)
    # /!\ [imshow: array is read as the images pixels: VIS(i,j), j->U, i->-V]
    Euv = 0.8
    U, V = np.linspace(-Euv, Euv, Nuv), np.linspace(Euv, -Euv, Nuv)
    VIS2 = np.zeros((Nuv, Nuv), dtype=np.float)
    for i in tqdm(range(Nuv)):
        for j in range(Nuv):
            VIS2[i, j] = np.abs(mod_VIS(zetac, rho, U[j], V[i], model, Nc=20000, LLD=0., Na=8, tol=1e-5))**2
    
    # rainbow colors
    Bcols = [cm.gist_rainbow(a) for a in np.linspace(0., 1., 6)]
    Bcols.reverse()
    
    # plot design
    if ax is None:
        plt.close('all')
        plt.subplots(figsize=(3.1, 3))
        plt.subplots_adjust(top = 0.96, bottom=0.16, left=0.22, right=0.98)
        ax = plt.subplot(1, 1, 1)

    # plot size and labels
    #    ax.set_title(r'Einstein $(u_{\rm E},v_{\rm E})$ plane')
    ax.set_xlabel(r'$u_{\rm E}$ $[\theta_{\rm E}^{-1}]\quad$ West $\longrightarrow$')
    ax.set_ylabel(r'$v_{\rm E}$ $[\theta_{\rm E}^{-1}]\quad$ North $\longrightarrow$')
    ax.set_xlim([0.7, -0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_xticks([-0.5, 0., 0.5])
    ax.set_yticks([-0.5, 0., 0.5])

    # plot theoretical visibility
    ax.imshow(VIS2, interpolation='quadric', cmap=cm.gray, vmin=0., vmax=1., extent=[-Euv, Euv, -Euv, Euv])

    # plot resolution contours: 0.5x, 1x, 1.5x typical resolution. Here, du = 0.5
    #    levels = 0.5 * np.array([0.25, 0.5, 0.75])
    levels = np.array([0.25, .5])
    ang = 2. * np.pi * np.linspace(0, 1, 100)
    lw = 0.8
    ax.plot([levels * np.cos(p) for p in ang], [levels * np.sin(p) for p in ang], 'w:', lw=lw)

    # conversion to mas
    mas = np.pi / 180. / 3600. / 1000.
    
    # compute and plot observed (u, v)
    if type(take) != list:
        take = [take]
    for takeit in take:
        for b in tqdm(range(6)):
            for l in range(6):
                uvdata = obs_VIS2(oifits, b + 1, l + 1, ref_MJD=ref_MJD)
                u, v, VIS2, VIS2ERR, DATE = uvdata[takeit]
                u = u * mas * thetaE
                v = v * mas * thetaE
                moduv = (u**2 + v**2)**0.5
                VISE2 = np.abs(mod_VIS(zetac, rho, u, v, model, Nc=20000, LLD=0., Na=8, tol=1e-5))**2
                Bcol = Bcols[l]

                # observed (u, v)
                for signe in [1., -1.]:
                    ax.scatter(signe * u, signe * v, marker='.', c=[Bcol], s=30)

    if figname is not None:
        plt.savefig(figname)


if __name__ == "__main__":

    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # choose visibility model
    #model = 'ESPL_PS_SINGEP'
    #model = 'ESPL_FULLINT_SINGEP'
    model = 'ESPL_FLAT_SINGEP'
    
    # best-fit parameters
    alpha, rho, thetaE, u0, tE, t0 = get_bestfit('mcmc_all_series_11.h5')

    # convert to East-right convention of the code
    alpha = 180. - alpha
    u0 = -u0

    # u,v plane sampling (default = 40)
    Nuv = 40
    
    # NB: pas de LLD dans le plot car ne se voit pas en pratique
    
    # July 12, take 1, take 2, take 1+2
    oifits = '2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits'

    fig_uv(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-12A.pdf')

    fig_uv(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-12B.pdf')

    fig_uv(oifits, [0, 1], model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-12AB.pdf')

    # July 19, take 1, take 2, take 1+2
    oifits = '2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits'

    fig_uv(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-19A.pdf')

    fig_uv(oifits, 1, model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-19B.pdf')

    fig_uv(oifits, [0, 1], model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-19AB.pdf')

    # July 21, take 1
    oifits = '2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits'

    fig_uv(oifits, 0, model, alpha, rho, thetaE, u0, tE, t0, Nuv, figname='uv-21A.pdf')


