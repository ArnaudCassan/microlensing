# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 14.4.21

import os, sys
import numpy as np
import emcee
from mcmc import InterferolensingModels
from utils import verbosity, printi, printd, printw

def get_bestfit(samplesfile):
    """Get best-fit parameters from input .h5 file.
    """
    # set I/O shell display
    tcol, tend, tit = "\033[0m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # check file
    if not os.path.isfile(samplesfile):
        raise ValueError("Input file missing.")
        
    # read PIONIER smapler file
    printi(tcol+"PIONIER all-epochs : "+tit+f"{samplesfile}"+tend)
    reader = emcee.backends.HDFBackend(samplesfile)
    
    # selected samples (keep all samples to get best fit)
    samples = reader.get_chain(discard=0, flat=True, thin=1)
    ndim = np.shape(samples)[1]
    log_prob_samples = reader.get_log_prob(discard=0, flat=True, thin=1)
    
    # convert to paper conventions /!\ but keep ref_MJD=8681.5 for t0'
    samples[:, 0] = (180. - samples[:, 0]) % 360.
    samples[:, 2] = -samples[:, 2]
    
    # check stats
    printi(f"  | flat chain shape : {samples.shape}")
    
    # best-fit parameters (alpha, thetaE, eta0, t*, t0)
    arg = np.argmax(log_prob_samples)
    bestfit = [samples[arg, i] for i in range(ndim)]
    alpha, thetaE, eta0, ts, t0 = bestfit
    
    # get model
    ilm = InterferolensingModels('ESPL_FLAT_MULTIEP')
    
    printi(tcol+"Best-fit parameters "+ilm.pnames+tend)
    printi("  | "+str(bestfit))
    
    bestchi2 = -2. * log_prob_samples[arg]
    printi("  | best-fit chi2 : "+str(bestchi2))
    
    # convert to microlensing parameters with 'fiducial' rho
    #    rho = 0.0314
    rho = 0.03198
    tE = ts / rho
    u0 = rho / eta0
    
    printi(tcol+"Microlensing parameters (alpha, rho, thetaE, u0, tE, t0)"+tend)
    printi("  | "+str([alpha, rho, thetaE, u0, tE, t0]))
        
    return alpha, rho, thetaE, u0, tE, t0


if __name__ == "__main__":

    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')

    get_bestfit('mcmc_all_series_11.h5')



#    alpha, thetaE, eta0, ts, t0 = 152.81391066943516, 0.7664500330895898, -1.6885042966612316, 3.6112199363763677, -0.39081199448418374

# Date conventions:
# DATE_PIONIER = MJD = JD - 2400000.5 : adopted convention
# DATE_ML = JD - 2450000 (= MJD + 2400000.5 - 2450000 = MJD - 49999.5)
# DATE_Gaia = JD


    
