# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 19.6.21, 12.6.21, 3.6.21, 27.5.21, 10/03/2021

import os, sys
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from astropy import units as u
from utils import verbosity, printi, printd, printw

def cal_physical(samplesfileP, samplesfileLC, DS, figref, burnin=0):
    """Compute lens mass and plot prob. distribution.
    
    Parameters
    ----------
    samplesfileP : string
        PIONIER all epochs MCMC chains.
    samplesfileLC : string
        Combined light curve + PIONIER all epochs MCMC chains.
    DS : tuple
        Source distance (value, error).
    figref : string
        Key string for output names of figures.
    """
    # set I/O shell display
    tcol, tend, tit = "\033[0m\033[36m", "\033[0m", "\033[0m\033[3m"
    
    # read PIONIER samples file
    printi(tcol+"  | PIONIER all-epochs : "+tit+f"{samplesfileP}"+tend)
    reader = emcee.backends.HDFBackend(samplesfileP)
    Psamples = reader.get_chain(flat=True, discard=burnin)
    
    # read light curve samples file, and use reference date 2458681
    printi(tcol+"  | LC combined fit parameters : "+tit+f"{samplesfileLC}"+tend)
    LCsamples = np.load(samplesfileLC)
    LCsamples[:, 0] = LCsamples[:, 0] - 2458681.

    # plot microlensing parameters correlations
    plt.close('all')
    corner.corner(LCsamples[:,0:6], labels=[r'$t_0$', r'$u_0$', r'$t_E$', r'$\rho$', r'$\pi_{{\rm E}, N}$', r'$\pi_{{\rm E}, E}$'], show_titles=True, color='dimgray', title_fmt='.5f')
    plt.savefig('corr_'+figref+'.pdf')
    
    # sampling of thE and piE distributions
    ns = 40000
    
    # DS
    printi(f"  | DS source distance [kpc]  \t {DS[0]} ± {DS[1]}")
        
    # thetaE
    med_thE = np.median(Psamples[:,1])
    sig_thE = np.std(Psamples[:,1])
    dist_thE = np.random.normal(med_thE, sig_thE, ns)
    printi(f"  | thE [mas]             \t {med_thE} ± {sig_thE}")
    
    # piE
    piE = np.sqrt(LCsamples[:,4].T**2 + LCsamples[:,5].T**2)
    med_piE = np.median(piE)
    sig_piE = np.std(piE)
    dist_piE = np.random.normal(med_piE, sig_piE, ns)
    printi(f"  | piE (median, sig)     \t {med_piE} ± {sig_piE}")
    
    # compute mass distribution
    kappa = 8.144
    dist_M = dist_thE / (kappa * dist_piE)
    med_M = np.median(dist_M)
    sig_M = np.std(dist_M)
    printi(f"  | lens mass [Mo]       \t {med_M} ± {sig_M}")
   
    # compute distances distributions
    dist_DS = np.random.normal(DS[0], DS[1], ns)
    dist_DL = 1. / (1. / dist_DS + dist_thE * dist_piE)
    med_DL = np.median(dist_DL)
    sig_DL = np.std(dist_DL)
    printi(f"  | DL lens distance [kpc] \t {med_DL} ± {sig_DL}")
    
    # tE
    tE = LCsamples[:,2].T
    med_tE = np.median(tE)
    sig_tE = np.std(tE)
    dist_tE = np.random.normal(med_tE, sig_tE, ns)
    printi(f"  | tE [days]           \t {med_tE} ± {sig_tE}")
    
    # rho
    rho = LCsamples[:,3].T
    med_rho = np.median(rho)
    sig_rho = np.std(rho)
    dist_rho = np.random.normal(med_rho, sig_rho, ns)
    printi(f"  | rho                 \t {med_rho} ± {sig_rho}")
    
    # murel
    dist_murel = dist_thE / dist_tE * 365.25
    med_murel = np.median(dist_murel)
    sig_murel = np.std(dist_murel)
    printi(f"  | murel [mas/year]    \t {med_murel} ± {sig_murel}")
    
    # thS
    dist_thS = dist_rho * dist_thE * 1000.
    med_thS = np.median(dist_thS)
    sig_thS = np.std(dist_thS)
    printi(f"  | thS [muas]          \t {med_thS} ± {sig_thS}")
    
    # R*
    conv = ((1e-6 * u.arcsec) * (1e3 * u.pc) / u.R_sun).si # / (u.R_sun).si
    dist_RS = dist_thS * dist_DS * conv.value
    med_RS = np.median(dist_RS)
    sig_RS = np.std(dist_RS)
    printi(f"  | RS [Ro] (pour info)  \t {med_RS} ± {sig_RS}")
    
#    # t*
#    dist_tS = dist_rho * dist_tE
#    med_tS = np.median(dist_tS)
#    sig_tS = np.std(dist_tS)
#    printi(f"  | t* [day] /!\ from LC \t {med_tS} ± {sig_tS}")
   
    # plot piE distribution
    plt.close('all')
    corner.corner(piE, labels=[r'$\pi_{\rm E}$'], show_titles=True, color='dimgray', title_fmt='.4f')
    plt.tight_layout()
    plt.savefig('piE_'+figref+'.pdf')
        
    # plot lens mass distribution
    plt.close('all')
    corner.corner(dist_M.T, labels=[r'$M$'], show_titles=True, color='dimgray', title_fmt='.3f')
    plt.tight_layout()
    plt.savefig('M_'+figref+'.pdf')
           
    # plot lens distance distribution
    plt.close('all')
    corner.corner(dist_DL.T, labels=[r'$D_L$'], show_titles=True, color='dimgray', title_fmt='.3f')
    plt.tight_layout()
    plt.savefig('DL_'+figref+'.pdf')
    
    # plot piEN/piEE
    plt.close('all')
    labels = [r'$\pi_{{\rm E},N} / \pi_{{\rm E},E}$']
    varpi = np.tan(np.deg2rad(Psamples[:, 0]))
    corner.corner(varpi, labels=labels, show_titles=True, color='dimgray', title_fmt='.3f')
    plt.tight_layout()
    plt.savefig('varpi_'+figref+'.pdf')
    
    
    # plot combined figure varpi, piE, thetaE
    plt.close('all')
    fig = plt.figure(figsize=(12, 3))
    
    xp, yp = -0.18, 1.03
    bins = 30
    
    ax = plt.subplot(1, 3, 1)
    ax.text(xp, yp, 'a', weight='bold', transform=ax.transAxes)
    ax.hist(varpi, bins=bins, histtype='step', density=True, color='dimgray')
    ax.set_xlabel(r'$\pi_{{\rm E},N} / \pi_{{\rm E},E}$')
    ax.set_ylabel('Probability density')

    ax = plt.subplot(1, 3, 2)
    ax.text(xp, yp, 'c', weight='bold', transform=ax.transAxes)
    ax.hist(piE, bins=bins, histtype='step', density=True, color='dimgray')
    ax.set_xlabel(r'$\pi_{\rm E}$')
    ax.set_ylabel('Probability density')
    
    ax = plt.subplot(1, 3, 3)
    ax.text(xp, yp, 'b', weight='bold', transform=ax.transAxes)
    ax.hist(dist_thE, bins=bins, histtype='step', density=True, color='dimgray')
    ax.set_xlabel(r'$\theta_{\rm E}$')
    ax.set_xticks([0.75, 0.76, 0.77, 0.78])
    ax.set_ylabel('Probability density')

    plt.tight_layout(w_pad=3.)

    plt.savefig('fig_varpiThEpiE.pdf')
    
    
    # plot combined figure pysical
    plt.close('all')
    fig = plt.figure(figsize=(9, 3))
    
    xp, yp = -0.18, 1.03
    bins = 30
    
    ax = plt.subplot(1, 2, 1)
    ax.text(xp, yp, 'a', weight='bold', transform=ax.transAxes)
    ax.hist(dist_M.T, bins=bins, histtype='step', density=True, color='dimgray')
    ax.set_xlabel(r'$M$ [M$_\odot$]')
    ax.set_ylabel('Probability density')
    
    ax = plt.subplot(1, 2, 2)
    ax.text(xp, yp, 'b', weight='bold', transform=ax.transAxes)
    ax.hist(dist_DL.T, bins=bins, histtype='step', density=True, color='dimgray')
    ax.set_xlabel(r'$D_L$ [kpc]')
    ax.set_ylabel('Probability density')

    plt.tight_layout(w_pad=3.)

    plt.savefig('fig_physical.pdf')


if __name__ == "__main__":
    
    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')
    
    # set I/O shell display
    tcol, tend = "\033[0m\033[34m", "\033[0m"
    
    # Source distance (mean, sigma)
    DS = (8.4, 1.5)#0.8 # en fait, barre legerement asymetrique
    
    # adapt font size
    plt.rc('font', size=11)
        
#    # murel photometry
#    printi(tcol + "Ground + Gaia :" + tend)
#    cal_physical('mcmc_all_series_11.h5', 'unconstrained.npy', DS, 'ground', burnin=100)

    # murel photometry + PIONIER
    printi(tcol + "Ground + Gaia + PIONIER :" + tend)
    cal_physical('mcmc_all_series_11.h5', 'PIONIER-constrained.npy', DS, 'combined', burnin=100)
    
    
#     0.755 ± 0.013
    
# 22.01.2021 :
#    printi(tcol + "Ground light curve only :" + tend)
#    cal_physical('mcmc_all_series_11.h5', 'samples_no-prior.npy', DS, 'ground', burnin=100)
#
#    printi(tcol + "Combined PIONIER + ground light curve :" + tend)
#    cal_physical('mcmc_all_series_11.h5', 'samples_PIONIER-prior.npy', DS, 'combined', burnin=100)
