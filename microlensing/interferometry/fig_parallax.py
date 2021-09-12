# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 14.4.21

import os, sys
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from utils import verbosity, printi, printd, printw
    
def fig_parallax(samplesfileP, samplesfileLCg, samplesfileLC, figname, burnin=0):
    """Plot parallax confidence ellipses.

    Parameters
    ----------
    samplesfileP : string
        PIONIER alone all epochs MCMC chains.
    samplesfileLCg : string
        Combined light curve alone MCMC chains.
    samplesfileLC : string
        Combined light curve + PIONIER all epochs MCMC chains.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # plot design
    plt.close('all')
    plt.subplots(figsize=(4.6, 4))
    plt.subplots_adjust(top = 0.96, bottom=0.12, left=0.14, right=0.98)
    ax = plt.subplot(1, 1, 1)
    
    ax.set_aspect('equal')
    ax.grid(ls='-', lw=0.3, which='both')
    
    ax.set_xlabel(r'$\pi_{{\rm E}, E}$ $[\theta_{\rm E}]\quad$ West $\longrightarrow$')
    ax.set_ylabel(r'$\pi_{{\rm E}, N}$ $[\theta_{\rm E}]\quad$ North $\longrightarrow$')

    ax.set_xlim([-0.052, -0.092])
    ax.set_ylim([-0.052, -0.012])
    
    ax.set_xticks([-0.09, -0.08, -0.07, -0.06])
    ax.set_yticks([-0.05, -0.04, -0.03, -0.02])
    
#    ax.set_xticks([-0.085, -0.075, -0.065, -0.055], minor=True)
#    ax.set_yticks([-0.045, -0.035, -0.025, -0.015], minor=True)

    # read PIONIER sampler file
    reader = emcee.backends.HDFBackend(samplesfileP)
    Psamples = reader.get_chain(flat=True, discard=burnin)
    
    # compute constrain on varpi=piEN/piEE from PIONIER alone
    varpi = np.tan(np.deg2rad(Psamples[:, 0]))
    med_varpi = np.median(varpi)
    sig_varpi = np.std(varpi)
    printi(tcol+"From PIONIER : \n"+tend+f"    piEN/piEE (median, sig) = {med_varpi}, {sig_varpi}")
    
    med = corner.quantile(varpi, 0.5)
    sigs = [0.00149, 0.0227, 0.1587, 0.8413, 0.9772, 0.99865]
    sigs_varpi = corner.quantile(varpi, sigs)
    
    X = np.array([0, -1])
    k1m, k1p, k2m, k2p, k3m, k3p = sigs_varpi[2], sigs_varpi[3], sigs_varpi[1], sigs_varpi[4], sigs_varpi[0], sigs_varpi[5]
    
    # ground light curve alone
    data = np.load(samplesfileLCg)
    piENg, piEEg = data[:,4:6].T
    
    # combined PIONIER + ground light curve
    data = np.load(samplesfileLC)
    piEN, piEE = data[:,4:6].T
    
    # common plot design
    lw = 0.3
    ec = 'k'
    alph = [0.8, 0.6, 0.4]
    kwargs = {'linewidth':lw, 'edgecolor':ec, 'fill':True}
    pkwargs = {'linewidth':lw, 'edgecolor':ec}

    # plot PIONIER varpi constraint
    ax.fill_between(X, np.array([0., -k1m]), np.array([0., -k1p]), fc='darkgray', alpha=alph[0], **pkwargs, zorder=20)
    ax.fill_between(X, np.array([0., -k2m]), np.array([0., -k2p]), fc='silver', alpha=alph[1], **pkwargs, zorder=18)
    ax.fill_between(X, np.array([0., -k3m]), np.array([0., -k3p]), fc='gainsboro', alpha=alph[2], **pkwargs, zorder=16)
    
    # plot ground light curve constraint
    confidence_ellipse(piEEg, piENg, ax, n_std=1, fc='deepskyblue', alpha=alph[0], **kwargs, zorder=10)
    confidence_ellipse(piEEg, piENg, ax, n_std=2, fc='skyblue', alpha=alph[1], **kwargs, zorder=8)
    confidence_ellipse(piEEg, piENg, ax, n_std=3, fc='powderblue', alpha=alph[2], **kwargs, zorder=6)

    # plot combined constraint
    confidence_ellipse(piEE, piEN, ax, n_std=1, fc='darkorange', alpha=alph[0], **kwargs, zorder=30)
    confidence_ellipse(piEE, piEN, ax, n_std=2, fc='orange', alpha=alph[1], **kwargs, zorder=28)
    confidence_ellipse(piEE, piEN, ax, n_std=3, fc='peachpuff', alpha=alph[2], **kwargs, zorder=26)
    
    # annotations
    ax.text(-0.072, -0.024, "Light curve", c='dodgerblue')
    ax.text(-0.053, -0.036, "PIONIER", c='dimgray')
    ax.text(-0.06, -0.044, "PIONIER \n+ light curve", c='darkorange', horizontalalignment='left')
        
    plt.savefig(figname)

def confidence_ellipse(x, y, ax, n_std=3.0, **kwargs):
    """Plot confidence ellipses.
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    # checks (mode debug)
    if n_std == 1:
        n = 4
        printd("Check piEN : {0} ± {1}".format(np.round(mean_y, n), np.round(scale_y, n)))
        printd("Check piEE : {0} ± {1}".format(np.round(mean_x, n), np.round(scale_x, n)))
    
    return ax.add_patch(ellipse)

if __name__ == '__main__':

    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # ground + Gaia / ground + Gaia + PIONIER
    fig_parallax('mcmc_all_series_11.h5', 'unconstrained.npy', 'PIONIER-constrained.npy', 'Cassan_Fig3.eps', burnin=100) # fig_parallax.pdf
   
# 7.12.20 : ground / ground + PIONIER
#    fig_parallax('mcmc_all_series_11.h5', 'samples_no-prior.npy', 'samples_PIONIER-prior.npy', 'fig_parallax.pdf', burnin=100)





