# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 27.5.21, 14.4.21

import sys, os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
from utils import verbosity, printi, printd, printw
from data import obs_T3PHI
from ESPL import vismag
from tqdm import tqdm
import fnmatch
from astropy.time import Time

def fig_lc(figname):
    """Plot magnification curve.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[32m", "\033[0m\033[1m\033[32m", "\033[0m", "\033[0m\033[3m"
    
    # infos on chosen data sets
    printi(tcol + "Including only ground-based data sets..." + tend)
    
    # get data sets
    dir = 'fig_lc_data/'
    datasets = fnmatch.filter(os.listdir(dir), '*.dat')
    
    datef, magf, errf = [], [], []
    for dset in datasets:
        print(tcol+"  loading data set: "+tend+f"{dset}")
        
        date, mag, err = np.loadtxt(dir+dset, unpack=True, delimiter=' ')
        
        datef.append(date - 2450000.)
        magf.append(mag)
        errf.append(err)
        
    # get theoretical ligh curve
    t, magth, _ = np.loadtxt(dir+'continuum_I.txt', unpack=True, delimiter=' ')

    # plot layout
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.subplots_adjust(top = 0.98, bottom=0.13, left=0.13, right=0.89)
    
    # plot limits
    X = [8672., 8691.]
    Y = [11.16, 8.9]
    ExtIH = 2.78
    
    # I-band axis
    ax.set_xlabel(r'$\rm{JD} - 2,450,000$')
    ax.set_ylabel(r'$I$-band magnitude')
    ax.set_xlim(X)
    ax.set_ylim(Y)
    ax.set_xticks([8675, 8680, 8685, 8690])
    ax.set_yticks([9, 9.5, 10, 10.5, 11])
    ax.xaxis.set_minor_locator(MultipleLocator(1.))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    
    # H-band axis
    ay = fig.add_axes(ax.get_position())
    ay.patch.set_visible(False)
    ay.yaxis.set_label_position('right')
    ay.yaxis.set_ticks_position('right')
    ay.axes.get_xaxis().set_visible(False)
    ay.set_ylabel(r'$H$-band magnitude')
    ay.set_xlim(X)
    ay.set_ylim(np.array(Y) - ExtIH)
    ay.set_yticks([6.5, 7, 7.5, 8])
    ay.yaxis.set_minor_locator(MultipleLocator(.1))
    
    # plot theoretical light curve
    ax.plot(t - 2450000, magth, 'k', lw=1.1)
    
    # plot data
    for i in range(len(datef)):
        ax.errorbar(datef[i], magf[i], errf[i], lw=0.9, zorder=20, fmt='o', ms=2.6, color='mediumblue')
    
    # get date of epoch and add marks
    ref_MJD = 58681.
    DATE = []
    phidata = obs_T3PHI('2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits', 1, 1, ref_MJD=ref_MJD)
    DATE.append(phidata[0][7] + ref_MJD - 49999.5)
    DATE.append(phidata[1][7] + ref_MJD - 49999.5)
    phidata = obs_T3PHI('2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits', 1, 1, ref_MJD=ref_MJD)
    DATE.append(phidata[0][7] + ref_MJD - 49999.5)
    DATE.append(phidata[1][7] + ref_MJD - 49999.5)
    phidata = obs_T3PHI('2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits', 1, 1, ref_MJD=ref_MJD)
    DATE.append(phidata[0][7] + ref_MJD - 49999.5)
    
    lw = 1.5
    shift = 0.1 + 0.5
    ref_mag = 8 + ExtIH + 0.46
    LT = -0.15 - shift
    L12 = -0.7 - shift
    L19 = -0.83 - shift
    L21 = -0.36 - shift
    size = 9
    yshift = -0.05 - 0.13
    tshift = -0.85
    
    col = 'red'#saddlebrown'#'tomato'#'orangered'#'cornflowerblue'
    
    # PIONIER limiting magnitudes
    ay.plot([8600, 8800], [7.5, 7.5], '--', c=col, lw=0.9)
    ay.text(8686.9, 7.5 - 0.05, r'PIONIER $H_{\rm lim}$', fontsize=size, c=col)
    
    col = 'black'#'mediumblue'
    
    # trigger
    ax.arrow(8674.8, ref_mag, 0, LT, length_includes_head=True, head_width=0.2, head_length=0.07, lw=lw, color=col)
    ax.text(8674.8 + tshift, ref_mag + yshift, 'Trigger', rotation=90, fontsize=size, c=col)
    
    # check dates
    printi(tun+"Check JD of takes"+tend)
    printi(f"  12 July, take 1 : JD-2450000 = {DATE[0]}")
    printi(f"  12 July, take 2 : JD-2450000 = {DATE[1]}")
    printi(f"  19 July, take 1 : JD-2450000 = {DATE[2]}")
    printi(f"  19 July, take 2 : JD-2450000 = {DATE[3]}")
    printi(f"  21 July, take 1 : JD-2450000 = {DATE[4]}")
    
    # July 12
    ax.arrow(DATE[0], ref_mag, 0, L12, length_includes_head=True, head_width=0.2, head_length=0.07, lw=lw, color=col)
    ax.text(DATE[0] + tshift, ref_mag + yshift, '12 July', rotation=90, fontsize=size, c=col)

    # July 19
    ax.arrow(DATE[3], ref_mag, 0, L19, length_includes_head=True, head_width=0.2, head_length=0.07, lw=lw, color=col)
    ax.text(DATE[3] + tshift, ref_mag + yshift, '19 July', rotation=90, fontsize=size, c=col)
    
    # July 21
    ax.arrow(DATE[4], ref_mag, 0, L21, length_includes_head=True, head_width=0.2, head_length=0.07, lw=lw, color=col)
    ax.text(DATE[4] + tshift, ref_mag + yshift, '21 July', rotation=90, fontsize=size, c=col)

    plt.savefig(figname)


if __name__ == "__main__":

    ## SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')
    
    fig_lc('Cassan_EDFig1.eps') # fig_lc.pdf




### oldies
#
#    # get data sets
##    dir = 'Gaia19bld_datasets/'
##    datasets = fnmatch.filter(os.listdir(dir), 'cleaned_*.dat')
#    dir = 'final_data/'
#    datasets = fnmatch.filter(os.listdir(dir), '*.dat')
#
#    datef, magf, errf = [], [], []
#    for dset in datasets:
#        date, mag, err = np.loadtxt(dir + dset, unpack=True, delimiter=' ')
#        if 'Gaia' in dset:
#            mag = mag - 1.3
#        excludelist = ['Gaia', 'ROAD', 'Spitzer', 'Kumeu', 'FCO']
#
#        # include only ground-based data sets
#        if not any([ex in dset for ex in excludelist]):
#            printi(tcol + "Including data set : " + tend + "{}".format(dset))
#            datef.append(date - 2450000)
#            magf.append(mag)
#            errf.append(err)
#
#    # get theoretical ligh curve
#    t, magth = np.loadtxt(dir+'model_LC.mod', unpack=True, delimiter=' ')
