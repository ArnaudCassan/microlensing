# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 30.6.21, 27.5.21, 9/12/2020

import sys
import numpy as np
import emcee
import corner
from accorner import accorner
import matplotlib.pylab as plt
from mcmc import InterferolensingModels
from utils import checkandtimeit, verbosity, printi, printd, printw

def fig_correl(samplesfile, model, figname, burnin=None, thin=None, dof=None, ref=''):
    """Plot MCMC results in corner plot.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # FOR PLOT ONLY : reference date in JD (not MJD) - 2458681
    ref_JD = 2458681.5 - 2458681.

    # get MCMC smapler HDF5 file
    reader = emcee.backends.HDFBackend(samplesfile)
    printi(" ")
    printi(tcol+"Sampler file : "+tit+f"{samplesfile}"+tend)

    # get model
    ilm = InterferolensingModels(model)
    labels = ilm.lxnames
    
    # try get auto-correlation or set manual input
    try:
        tau = reader.get_autocorr_time()
    except emcee.autocorr.AutocorrError:
        tau = None
    
    if burnin is None:
        if tau is not None: burnin = int(2*np.max(tau))
        else: burnin = 0
    if thin is None:
        if tau is not None: thin = int(0.5*np.min(tau))
        else: thin = 1
    
    # get final samples (removing bun-in phase)
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    ndim = np.shape(samples)[1]
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

    # change parameters to fit microlensing conventions
    if model == 'ESPL_FLAT_SINGEP':
        nojump = 20. # to avoid 0-360 jump
        samples[:, 0] = (90 - samples[:, 0] + nojump) % 360. - nojump
        
    if model == 'ESPL_FLAT_MULTIEP' or model == 'ESPL_FLAT_MULTIEP_RESC':
        samples[:, 0] = (180. - samples[:, 0]) % 360.
        samples[:, 2] = -samples[:, 2]
        samples[:, 4] = ref_JD + samples[:, 4]
        
    if model == 'ESPL_MLENSING_MULTIEP':
        samples[:, 0] = (180. - samples[:, 0]) % 360.
        samples[:, 5] = -samples[:, 5]
        samples[:, 4] = ref_JD + samples[:, 4]

    # print stats on outputs
    printi("  | Burn-in : "+tit+f"{burnin}"+tend)
    printi("  | Thin : "+tit+f"{thin}"+tend)
    printi("  | Flat chain shape : "+tit+f"{samples.shape}"+tend)
 
    # best-fit
    arg = np.argmax(log_prob_samples)
    bestfit = [samples[arg, i] for i in range(ndim)]
    
    printi(tcol+"Best-fit parameters "+ilm.pnames+tend)
    printi("  | "+str(bestfit))
    
    bestchi2 = -2. * log_prob_samples[arg]
    printi("  | Best-fit chi2 : "+tend+str(bestchi2))
    
    if dof:
        bestchi2dof = bestchi2 / dof
        printi("  | Best-fit chi2/dof : "+tend+str(bestchi2dof))
    
    # medians
    meds = [corner.quantile(samples[:, i], [0.5])[0] for i in range(ndim)]
    printi(tcol+"Median parameters "+ilm.pnames+tend)
    printi("  | "+str(meds))
    
    printi(" ")

    # create plot
    plt.close('all')
    
    corner.corner(samples, labels=labels, show_titles=True, color='dimgray', title_fmt='.4f', plot_datapoints=False)
    
    ax = plt.gca()
    ax.text(-2.55, 3., ref, size='xx-large', weight='bold', transform=ax.transAxes)
        
    plt.savefig(figname)
    
    return samples

def fig_correl_all(model, jul12, jul19, jul21, figname):
    """Plot panels 12, 19 and 21 July in the same plot.
    """
    # get model
    ilm = InterferolensingModels(model)
    labels = ilm.lxnames
    
    # open figure
    plt.close('all')
    fig = plt.figure(figsize=(8,26)) # 4, 16
    
    # font size
    fs = 14 #8
    fst = 16
    # label pad
    lp = 0.05 #0.1
    
    plt.rc('font', size=fs)
    
    # create subplots
    axes = fig.subplots(11, 3)
    
    # plot 12 July
    accorner(jul12, labels=labels, title_kwargs={'size':fst}, labelpad=lp, show_titles=True, color='dimgray', title_fmt='.4f', axes=axes[0:3], plot_datapoints=False)
    
    # space
    [axes[3][i].remove() for i in range(3)]
    
    # plot 19 July
    accorner(jul19, labels=labels, title_kwargs={'size':fst}, labelpad=lp, show_titles=True, color='dimgray', title_fmt='.4f', axes=axes[4:7], plot_datapoints=False)

    # space
    [axes[7][i].remove() for i in range(3)]
        
    # plot 21 July
    accorner(jul21, labels=labels, title_kwargs={'size':fst}, labelpad=lp, show_titles=True, color='dimgray', title_fmt='.4f', axes=axes[8:11], plot_datapoints=False)
    
    # label pannels
    xpos = -2.55
    
    ax = axes[2][2]
    ax.text(xpos, 3., 'a', size='xx-large', weight='bold', transform=ax.transAxes)
    
    ax = axes[6][2]
    ax.text(xpos, 3., 'b', size='xx-large', weight='bold', transform=ax.transAxes)
    
    ax = axes[10][2]
    ax.text(xpos, 3., 'c', size='xx-large', weight='bold', transform=ax.transAxes)
    
    # pad panels
    pad = 0.05
    fig.subplots_adjust(hspace=pad, wspace=pad)
    
    # save figure
    plt.savefig(figname, bbox_inches='tight')
    
def comp_thetaE(samples12, samples19, samples21, corr='', syst='', ref='', bins=30, show=True, figname=None):
    """Compare thetaE between 12-19-21 epochs.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # plot design
    plt.close('all')
    plt.figure(figsize=(7, 4))
    
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel(r'$\theta_{\rm E}$ [mas]')
    ax.set_ylabel(r'Probability density')
    ax.set_xlim([0.7, 0.88])
    #    ax.set_ylim([0., 120])
    
    ax.text(-0.1, 1., ref, weight='bold', transform=ax.transAxes)
        
    labels = [[samples12, 'blue', 'July 12'], [samples19, 'red', 'July 19'], [samples21, 'green', 'July 21']]
    
    sigs = [0.00149, 0.0227, 0.1587, 0.8413, 0.9772, 0.99865]
    
    for (px, c, title) in labels:
    
        n, _, _ = ax.hist(px[:, 1].flatten(), bins=bins, histtype='step', density=True)

        sig1m, sig1p  = corner.quantile(px[:, 1], [sigs[2], sigs[3]])
        sig2m, sig2p  = corner.quantile(px[:, 1], [sigs[1], sigs[4]])
        
        sigpos = np.amax(n) * np.exp(-0.5) / (sig1m * np.sqrt(2. * np.pi))

        ax.plot([sig1m, sig1p], [sigpos, sigpos], c=c, lw=4)
        ax.plot([sig2m, sig2p], [sigpos, sigpos], c=c, lw=2)
        
        ax.text(0.72, sigpos, title, c=c) #, weight='bold')
        
    if show:
        ax.text(0.38, 0.9, r'Spectral channels correlation : '+corr, transform=ax.transAxes)
        ax.text(0.38, 0.8, r'Systematic error in calibration : '+syst, transform=ax.transAxes)
        
    plt.tight_layout()
    plt.savefig(figname)


if __name__ == '__main__':
    
    ## SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # individual 12, 19, 21 ; LLD=0.45 ; resc=(1., 0.03) ; corr=50% ; angle alpha 1 (3, 120, 130)
    plt.rc('font', size=14)
    samples12 = fig_correl('mcmc_12_series_11.h5', 'ESPL_FLAT_SINGEP', 'corr_12_alpha_1.pdf', ref='a', thin=1, burnin=70)
    samples19 = fig_correl('mcmc_19_series_11.h5', 'ESPL_FLAT_SINGEP', 'corr_19_alpha_1.pdf', ref='b', thin=1)
    samples21 = fig_correl('mcmc_21_series_11.h5', 'ESPL_FLAT_SINGEP', 'corr_21_alpha_1.pdf', ref='c', thin=1)
    
    comp_thetaE(samples12, samples19, samples21, corr='50%', syst='yes', show=False, bins=30, figname='comp_thetaE.pdf')
    comp_thetaE(samples12, samples19, samples21, corr='50%', syst='yes', bins=30, ref='b', figname='comp_thetaE_50.pdf')
    
    fig_correl_all('ESPL_FLAT_SINGEP', samples12, samples19, samples21, 'Cassan_EDFig5.eps') # fig_correl.pdf
    
    # combined 12+19+21, ; LLD=0.45 ; resc=(1., 0.03) ; corr=50% ; PIONIER alone ; arcs parameters
    plt.rc('font', size=12)
    fig_correl('mcmc_all_series_11.h5', 'ESPL_FLAT_MULTIEP', 'Cassan_EDFig6.eps', thin=1, burnin=100) # corr_all.pdf
    
#    # individual 12, 19, 21 ; LLD=0.45 ; resc=(1., 0.03) ; corr=50% ; angle alpha 2 (183, 300, 310)
#    plt.rc('font', size=14)
#    fig_correl('mcmc_12_series_13.h5', 'ESPL_FLAT_SINGEP', 'corr_12_alpha_2.pdf', thin=1, burnin=70)
#    fig_correl('mcmc_19_series_13.h5', 'ESPL_FLAT_SINGEP', 'corr_19_alpha_2.pdf', thin=1)
#    fig_correl('mcmc_21_series_13.h5', 'ESPL_FLAT_SINGEP', 'corr_21_alpha_2.pdf', thin=1)

#    # individual 12, 19, 21 ; LLD=0.45 ; resc=(1., 0.03) ; corr=80% ; angle alpha 2 (183, 300, 310)
#    plt.rc('font', size=14)
#    samples12 = fig_correl('mcmc_12_series_10.h5', 'ESPL_FLAT_SINGEP', 'corr_12_series_10.pdf', thin=1, burnin=70)
#    samples19 = fig_correl('mcmc_19_series_10.h5', 'ESPL_FLAT_SINGEP', 'corr_19_series_10.pdf', thin=1)
#    samples21 = fig_correl('mcmc_21_series_10.h5', 'ESPL_FLAT_SINGEP', 'corr_21_series_10.pdf', thin=1)
#    comp_thetaE(samples12, samples19, samples21, corr='80%', syst='yes', bins=30, ref='', figname='comp_thetaE_80.pdf')

#    # individual 12, 19, 21 ; LLD=0.45 ; resc=(1., 0.03) ; corr=20% ; angle alpha 2 (183, 300, 310)
#    plt.rc('font', size=14)
#    samples12 = fig_correl('mcmc_12_series_9.h5', 'ESPL_FLAT_SINGEP', 'corr_12_series_9.pdf', thin=1, burnin=70)
#    samples19 = fig_correl('mcmc_19_series_9.h5', 'ESPL_FLAT_SINGEP', 'corr_19_series_9.pdf', thin=1)
#    samples21 = fig_correl('mcmc_21_series_9.h5', 'ESPL_FLAT_SINGEP', 'corr_21_series_9.pdf', thin=1)
#    comp_thetaE(samples12, samples19, samples21, corr='20%', syst='yes', bins=30, ref='', figname='comp_thetaE_20.pdf')

#    # individual 12, 19, 21 ; LLD=0.45 ; resc=(1., 0.) ; corr=0% ; angle alpha 2 (183, 300, 310)
#    plt.rc('font', size=14)
#    samples12 = fig_correl('mcmc_12_series_12.h5', 'ESPL_FLAT_SINGEP', 'corr_12_series_12.pdf', thin=1, burnin=70)
#    samples19 = fig_correl('mcmc_19_series_12.h5', 'ESPL_FLAT_SINGEP', 'corr_19_series_12.pdf', thin=1, burnin=60)
#    samples21 = fig_correl('mcmc_21_series_12.h5', 'ESPL_FLAT_SINGEP', 'corr_21_series_12.pdf', thin=1)
#    comp_thetaE(samples12, samples19, samples21, corr='0%', syst='no', bins=30, ref='a', figname='comp_thetaE_0.pdf')

#    # combined 12+19+21, ; LLD=0.45 ; resc=(1., 0.03) ; corr=50% ; PIONIER+prior on rho ; microlensing parameters
#    plt.rc('font', size=12)
#    fig_correl('mcmc_all_mlensing_series_11.h5', 'ESPL_MLENSING_MULTIEP', 'corr_all_mlensing.pdf', thin=1, burnin=100)




