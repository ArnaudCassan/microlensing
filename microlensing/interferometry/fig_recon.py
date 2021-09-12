# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 14.8.21, 30.6.21, 28.5.21, 5/12/2020

import sys
import numpy as np
import matplotlib.pylab as plt
from utils import verbosity, printi, printd, printw
from astropy.io import fits
from scipy import stats
from ESPL import newz
from data import obs_VIS2
from get_bestfit import get_bestfit

def fig_recon(oifits, alpha, rho, thetaE, u0, tE, t0, image, shift, figname=None, ax=None):
    """Images reconstruction.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

    # reference date (MJD)
    ref_MJD = 58681.

    # get date of take
    uvdata = obs_VIS2(oifits, 1, 1, ref_MJD=ref_MJD)
    DATE_ref = uvdata[0][4]

    # create rotated trajectory
    zetac = np.complex((DATE_ref - t0) / tE, u0) * np.exp(np.complex(0., np.pi / 180. * alpha))
    angle = 180 - np.rad2deg(np.angle(zetac)) % 360.
    printi(tcol + "Rotation angle of images (°) : " + tit + str(angle) + tend)
    
    # for the convention in the plot
    angle = 90 - angle
    
    # load SQUEEZE results
    img = fits.getdata(image)[0]
    hdr = fits.getheader(image)
    xmax = hdr['SCALE'] * img.shape[0]/2
    
    # define plot size
    if ax is None:
        plt.close('all')
        plt.subplots(figsize=(4, 3.7))
        plt.subplots_adjust(top = 0.98, bottom=0.15, left=0.21, right=0.98)
        ax = plt.subplot(1, 1, 1)
    
    # plot size and labels
    ax.set_xlabel(r'$x$ [mas]$\quad$ West $\longrightarrow$')
    ax.set_ylabel(r'$y$ [mas]$\quad$ North $\longrightarrow$')
    
    # plot limts
    theta_E = 0.766
    b = theta_E * 1.5
    ax.set_ylim([-b, b])
    ax.set_xlim([b, -b])
    
    # draw line angle alpha
    lw = 1.2
    x = np.linspace(-2., 2., 2)
    y1 = np.tan(np.deg2rad(angle - 90.)) * x
    ax.plot(x, y1, c='w', lw=lw, ls='--')
    ax.scatter([0.], [0.], marker='o', c='w', s=20)
    
    #    # plot Einstein ring
    #    ang = 2. * np.pi * np.linspace(0, 1, 100)
    #    ax.plot([theta_E * np.cos(p) for p in ang], [theta_E * np.sin(p) for p in ang], c='w', ls=':', lw=lw)

    # images
    nzp, nzm = newz(zetac, rho, 20000, 2000)
    nzp[-1], nzm[-1] = nzp[0], nzm[0]
    nxp, nyp = np.real(nzp), np.imag(nzp)
    nxm, nym = np.real(nzm), np.imag(nzm)
    ax.fill_between(theta_E * nxp, theta_E * nyp, facecolor='w', zorder=20)
    ax.fill_between(theta_E * nxm, theta_E * nym, facecolor='w', zorder=20)
        
    # plot image
    max = np.amax(img)
    ax.imshow(img, extent=(-xmax-shift[0],xmax-shift[0],-xmax-shift[1],xmax-shift[1]), cmap='gist_heat', vmin=0., vmax=1.07 * max)
    
    if figname is not None:
        plt.savefig(figname)

def fig_reconcombined(ois, alpha, rho, thetaE, u0, tE, t0, figname=None):
    """Combined figure.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

    # combined plot design
    plt.close('all')
    plt.figure(figsize=(12, 3.7))
    plt.subplots_adjust(top = 0.99, bottom=0.12, left=0.08, right=0.99, wspace=0.4)
    
    # fig ref pos / date
    xp, yp = -0.18, 0.99
    xd, yd = 0.06, 0.06
    
    # July 12, take 1+2
    ax = plt.subplot(1, 3, 1)
    ax.text(xp, yp, 'a', weight='bold', transform=ax.transAxes)
    ax.text(xd, yd, 'July 12', transform=ax.transAxes, c='white')
    ax = fig_recon(ois[0][0], alpha, rho, thetaE, u0, tE, t0, ois[0][1], ois[0][2], ax=ax)
    
    # July 19, take 1+2
    ax = plt.subplot(1, 3, 2)
    ax.text(xp, yp, 'b', weight='bold', transform=ax.transAxes)
    ax.text(xd, yd, 'July 19', transform=ax.transAxes, c='white')
    ax = fig_recon(ois[1][0], alpha, rho, thetaE, u0, tE, t0, ois[1][1], ois[1][2], ax=ax)
    
    # July 21, take 1
    ax = plt.subplot(1, 3, 3)
    ax.text(xp, yp, 'c', weight='bold', transform=ax.transAxes)
    ax.text(xd, yd, 'July 21', transform=ax.transAxes, c='white')
    ax = fig_recon(ois[2][0], alpha, rho, thetaE, u0, tE, t0, ois[2][1], ois[2][2], ax=ax)
    
    plt.savefig(figname)
    
    
if __name__ == '__main__':

    ## SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    # best-fit parameters
    alpha, rho, thetaE, u0, tE, t0 = get_bestfit('mcmc_all_series_11.h5')
        
    # convert to East-right convention of the code
    alpha = 180. - alpha
    u0 = -u0

    # adapt font size
    plt.rc('font', size=13)
    
    # files
    ois = [
    ('2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits', '2019-07-12_SCI_Gaia19lbd_image.fits', (0.02, -0.1)),
    ('2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits', '2019-07-19_SCI_Gaia19lbd_image.fits', (0.04, -0.16)),
    ('2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits', '2019-07-21_SCI_Gaia19lbd_image.fits', (0.02, -0.15))]
    
    # combined plot
    fig_reconcombined(ois, alpha, rho, thetaE, u0, tE, t0, figname='Cassan_Fig1.eps') # fig_recon.pdf

    # individual epochs
    fig_recon(ois[0][0], alpha, rho, thetaE, u0, tE, t0, ois[0][1], ois[0][2], figname='fig_recon_12.pdf')
    fig_recon(ois[1][0], alpha, rho, thetaE, u0, tE, t0, ois[1][1], ois[1][2], figname='fig_recon_19.pdf')
    fig_recon(ois[2][0], alpha, rho, thetaE, u0, tE, t0, ois[2][1], ois[2][2], figname='fig_recon_21.pdf')
    
    
#    # get date of epochs
#    ref_MJD = 58681.
#    phidata = obs_VIS2('2019-07-12_SCI_Gaia19lbd_oidataCalibrated.fits', 1, 1, ref_MJD=ref_MJD)
#    print("Epoch 12 July (JD-2450000): ", phidata[0][7] + ref_MJD - 49999.5, phidata[1][7] + ref_MJD - 49999.5)
#    phidata = obs_VIS2('2019-07-19_SCI_Gaia19lbd_oidataCalibrated.fits', 1, 1, ref_MJD=ref_MJD)
#    print("Epoch 19 July (JD-2450000): ", phidata[0][7] + ref_MJD - 49999.5, phidata[1][7] + ref_MJD - 49999.5)
#    phidata = obs_VIS2('2019-07-21_SCI_Gaia19lbd_oidataCalibrated.fits', 1, 1, ref_MJD=ref_MJD)
#    print("Epoch 21 July (JD-2450000): ", phidata[0][7] + ref_MJD - 49999.5)


