# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 19.6.21, 12.6.21

import os, sys
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from astropy import units as u
from utils import verbosity, printi, printd, printw

def calc_murel(thE, sig_thE, tE, sig_tE):
    """Compute murel for the three values A, B, C of the spectroscopy paper.
    """
    # set I/O shell display
    tcol, tend, tit = "\033[0m\033[36m", "\033[0m", "\033[0m\033[3m"
    
    printi(tcol + "theta_E(Ground + Gaia + Spitzer + Spectro) :" + tend + f" {thE} ± {sig_thE}")
   
    # sampling of thE and piE distributions
    ns = 40000
    
    # murel
    dist_thE = np.random.normal(thE, sig_thE, ns)
    dist_tE = np.random.normal(tE, sig_tE, ns)
    
    dist_murel = dist_thE / dist_tE * 365.25
    med_murel = np.median(dist_murel)
    sig_murel = np.std(dist_murel)
    printi(f"  | murel [mas/year]  {med_murel} ± {sig_murel}")

def calc_M(thE, sig_thE, piE, sig_piE):
    """Compute M for the three values A, B, C of the spectroscopy paper.
    """
    # sampling of thE and piE distributions
    ns = 40000
    
    # compute mass distribution
    dist_thE = np.random.normal(thE, sig_thE, ns)
    dist_piE = np.random.normal(piE, sig_piE, ns)
    kappa = 8.144
    dist_M = dist_thE / (kappa * dist_piE)
    med_M = np.median(dist_M)
    sig_M = np.std(dist_M)
    printi(f"  | lens mass [Mo]    {med_M} ± {sig_M}")


if __name__ == "__main__":
    
    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')
    
    # set I/O shell display
    tcol, tend, tit = "\033[0m\033[36m", "\033[0m", "\033[0m\033[3m"
    
    # Compute murel and M for the spectro A, B, C
    tE, sig_tE = 107.06, 0.50 # from photometry paper
    piE, sig_piE = 0.0823, 0.0018 # from photometry paper
    
    printi(tcol + "pi_E(Ground + Gaia + Spitzer + Spectro) :" + tend + f" {piE} ± {sig_piE}")
    printi(tcol + "t_E(Ground + Gaia + Spitzer + Spectro) :" + tend + f" {tE} ± {sig_tE}")
    
    thE, sig_thE = 0.754, 0.013
    calc_murel(thE, sig_thE, tE, sig_tE)
    calc_M(thE, sig_thE, piE, sig_piE)
   
    thE, sig_thE = 0.724, 0.012
    calc_murel(thE, sig_thE, tE, sig_tE)
    calc_M(thE, sig_thE, piE, sig_piE)
    
    thE, sig_thE = 0.721, 0.018
    calc_murel(thE, sig_thE, tE, sig_tE)
    calc_M(thE, sig_thE, piE, sig_piE)
    
    # pour check
    piE, sig_piE = 0.0818, 0.0020 # from PIONIER paper
    thE, sig_thE = 0.765, 0.0038 # from PIONIER paper
    printi(tcol + "pi_E(PIONIER + LC) :" + tend + f" {piE} ± {sig_piE}")
    printi(tcol + "t_E(LC) :" + tend + f" {tE} ± {sig_tE}")
    calc_murel(thE, sig_thE, tE, sig_tE)
    calc_M(thE, sig_thE, piE, sig_piE)
   
