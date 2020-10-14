# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import h5py
import pandas as pd
import subprocess
from itertools import repeat
from multiprocessing import Pool
from microlensing.utils import checkandtimeit, verbosity, printi, printd, printw
from microlensing.mismap.magclightc import MagnificationCurve, LightCurve

def process_results(gridsprefix, fitsprefix, nmod=9):
    """Process fit results
        
        nmod : int
            Maximal number of output models.
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[35m", "\033[0m\033[1m\033[35m", "\033[0m", "\033[0m\033[3m"

    # check that names do not have extension
    if '.hdf5' in gridsprefix:
        raise NameError("grid prefix should not contain .hdf5 extension")
    if '.hdf5' in fitsprefix:
        raise NameError("fit prefix should not contain .hdf5 extension")

    # check that mcgrid(.hdf5) exists
    grid = gridsprefix + '.hdf5'
    if not os.path.isfile(grid):
        raise IOError("file '" + grid + "' is missing")

    # verbose
    printd(tcol + "Grid file " + tit + "'" + grid + "'" + tend)

    # collect fit files and fill missing ones
    missingfits, missing = [], []
    with h5py.File(grid, 'r') as fgrid:
        Ngs = fgrid.attrs['Ngs']
        Ngq = fgrid.attrs['Ngq']
        k = 0
        Cgs, Cgq, Cgu0, Cgalpha, CgtE, Cgt0, Cgrho, Cgchidof, Cgchi = [],[],[],[],[],[],[],[],[]
        for j in range(Ngq):
            Lgs, Lgq, Lgu0, Lgalpha, LgtE, Lgt0, Lgrho, Lgchidof, Lgchi = [],[],[],[],[],[],[],[],[]
            for i in range(Ngs):
                gridfitsk = fitsprefix + '_' + str(k) + '.hdf5'
                if os.path.isfile(gridfitsk):
                    # fit file exists
                    with h5py.File(gridfitsk, 'r') as fgridfitsk:
                        Lgs.append(fgridfitsk['s'][:])
                        Lgq.append(fgridfitsk['q'][:])
                        Lgu0.append(fgridfitsk['u0'][:])
                        Lgalpha.append(fgridfitsk['alpha'][:])
                        LgtE.append(fgridfitsk['tE'][:])
                        Lgt0.append(fgridfitsk['t0'][:])
                        Lgrho.append(fgridfitsk['rho'][:])
                        Lgchidof.append(fgridfitsk['chidof'][:])
                        Lgchi.append(fgridfitsk['chi'][:])
                        fgridfitsk.flush()
                        fgridfitsk.close()
                else:
                    # fit file is missing
                    default = fgrid[str(i) + ' ' + str(j)]
                    meshs, meshq = np.meshgrid(default['s'][:], default['q'][:])
                    Lgs.append(meshs)
                    Lgq.append(meshq)
                    fails = np.full_like(meshs, np.inf)
                    Lgu0.append(fails)
                    Lgalpha.append(fails)
                    LgtE.append(fails)
                    Lgt0.append(fails)
                    Lgrho.append(fails)
                    Lgchidof.append(fails)
                    Lgchi.append(fails)
                    missingfits.append(gridfitsk)
                    missing.append((default['s'][:], default['q'][:]))
                k += 1
            Cgs.append(np.concatenate(Lgs, axis=1))
            Cgq.append(np.concatenate(Lgq, axis=1))
            Cgu0.append(np.concatenate(Lgu0, axis=1))
            Cgalpha.append(np.concatenate(Lgalpha, axis=1))
            CgtE.append(np.concatenate(LgtE, axis=1))
            Cgt0.append(np.concatenate(Lgt0, axis=1))
            Cgrho.append(np.concatenate(Lgrho, axis=1))
            Cgchidof.append(np.concatenate(Lgchidof, axis=1))
            Cgchi.append(np.concatenate(Lgchi, axis=1))
        fgrid.flush()
        fgrid.close()
    s = np.concatenate(Cgs, axis=0)
    q = np.concatenate(Cgq, axis=0)
    u0 = np.concatenate(Cgu0, axis=0)
    alpha = np.concatenate(Cgalpha, axis=0)
    tE = np.concatenate(CgtE, axis=0)
    t0 = np.concatenate(Cgt0, axis=0)
    rho = np.concatenate(Cgrho, axis=0)
    chidof = np.concatenate(Cgchidof, axis=0)
    chi = np.concatenate(Cgchi, axis=0)

    search_map = [s, q, chidof, chi, missing]

    # verbose
    if missingfits:
        printi(tcol + "Fit crashed for " + tit + str(len(missingfits)) + tcol + " sub-grids" + tend)
        for mi in missingfits:
            printd(tit + "  ('" +  mi + "')" + tend)

    # order models by X^2
    ind = np.unravel_index(np.argsort(chidof, axis=None), chidof.shape)

    models = list()
    for i in range(nmod):
        params = [u0[ind][i], alpha[ind][i], tE[ind][i], t0[ind][i], rho[ind][i]]
        if np.any(np.isinf(params)):
            nmod = i
            break
        models.append({'s': s[ind][i], 'q': q[ind][i], 'u0': u0[ind][i], 'alpha': alpha[ind][i], 'tE': tE[ind][i], 't0': t0[ind][i], 'rho':  rho[ind][i]})

    # list best-fit parameters
    befi = "  {0:<2s} {1:<9s} {2:<11s} {3:<12s} {4:<10s} {5:<10s} {6:<12s} {7:<11s} {8:<10s} {9:<12s}\n".format('', 's', 'q', 'u0', 'alpha', 'tE', 't0', 'rho', 'X^2/dof', 'X^2')
    for i in range(nmod):
        befi += "  {0:<2d} {1:<9.6f} {2:<11.4e} {3:<+12.4e} {4:<10f} {5:<10f} {6:<12f} {7:<11.4e} {8:<10f} {9:<12.1f}\n".format(i + 1, s[ind][i], q[ind][i], u0[ind][i], alpha[ind][i], tE[ind][i], t0[ind][i], rho[ind][i], chidof[ind][i], chi[ind][i])

    # create _rank.txt output file
    f = open(fitsprefix + '_rank.txt', 'w')
    f.write(befi)
    f.close()

    # verbose
    printi(tcol + "Best-fit models ranking:\n" + tend + befi)

    return search_map, models

def plot_search_map(search_map, models, figname=None, title=None):
    """Plot binary-lens X^2 search map"""
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[36m", "\033[0m\033[1m\033[36m", "\033[0m", "\033[0m\033[3m"
    
    # get search_map parameters
    s, q, chidof, chi, missing = search_map
    
    # replace ∞ X^2 grid points with max of X^2
    arg = np.where(chidof == np.inf)
    chidof[arg] = -1.
    chidof[arg] = np.unique(np.max(chidof))
    
    # interpolate and prepare grid (zfact x msize = 128)
    sizs = int(np.shape(chidof)[1])
    sizq = int(np.shape(chidof)[0])
    msize = np.max([sizs, sizq])
    gridsize = 24
    zfact = 128. / msize
    
    # plot values
    zchidof = 10 ** ndimage.zoom(np.log10(chidof), zfact)
    zs = 10 ** ndimage.zoom(np.log10(s), zfact)
    zq = 10 ** ndimage.zoom(np.log10(q), zfact)

    # plot laytout
    plt.close('all')
    plt.rc('font', size=12)
    fig, MAP = plt.subplots(1, figsize=(8,6))
    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.98, top=0.94, wspace=None, hspace=None)
    MAP.set_xscale('log')
    MAP.set_yscale('log')
    MAP.set_xlim([0.2, 5.])
    MAP.set_ylim([1e-5, 1.])
    MAP.set_xlabel(r'$s$')
    MAP.set_ylabel(r'$q$')
    if title:
        MAP.set_title(title)

    # create hex plot
    x = zs.ravel()
    y = zq.ravel()
    z = zchidof.ravel()
    hb = plt.hexbin(x, y, C=z, cmap=plt.cm.coolwarm, xscale='log', yscale='log', bins=None, gridsize=gridsize)
    cb = plt.colorbar(format='%.1f', boundaries=np.linspace(hb.norm.vmin, hb.norm.vmax, 100))
    cb.set_label(r'$\chi^2$/dof')
    plt.xticks(np.array([0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5]), np.array(['0.2', '0.3', '0.5', '0.7', '1', '2', '3', '4', '5']))

    # biff missing grid points
    if missing:
        dlogs = 0.5 * (np.log10(missing[0][0][1]) - np.log10(missing[0][0][0]))
        dlogq = 0.5 * (np.log10(missing[0][1][1]) - np.log10(missing[0][1][0]))
        for miss in missing:
            smin = 10 ** (np.log10(np.min(miss[0])) - dlogs)
            smax = 10 ** (np.log10(np.max(miss[0])) + dlogs)
            qmin = 10 ** (np.log10(np.min(miss[1])) - dlogq)
            qmax = 10 ** (np.log10(np.max(miss[1])) + dlogq)
            MAP.fill([smin, smax, smax, smin], [qmin, qmin, qmax, qmax], fill=False, hatch='x', lw=0)

    # mark best fits
    i = 1
    for model in models:
        plt.text(model['s'], model['q'], str(i), fontsize=6, color='k', ha='center', va='center', bbox=dict(facecolor='white', boxstyle='circle', alpha=1.))
        i += 1

    # save plot
    if figname:
        plt.savefig(figname)
    else:
        plt.show()

def create_maingrid(gridsprefix, srange, qrange, majgrid, mingrid, nmc, pcaus, axis=[0.1, 10., 8e-6, 1.2]):
    """Generate an HDF5 file with definition of sub- s,q-grids
        
        Parameters
        ----------
        gridsprefix : str
            Name (without any extension) of output HDF5 library of magnifications
            curves s,q-grids and corresponding PDF map of grids.
        srange : tuple
            Global range in s. Default is: srange=(0.2, 5.0)
        qrange : tuple
            Global range in q. Default is: qrange=(1e-5, 1.)
        majgrid : tuple
            Number of sub- s,q-grids. Default is: majgrid=(12, 5)
        mingrid : tuple
            Size of sub s,q-grids. Default is: mingrid=(7, 7)
        axis : float 1-D array, optional
            Plot limits. Usage: axis=[xmin, xmax, ymin, ymax].
            Default is: axis=[0.1, 10., 8e-6, 1.].
        plot : bool, optional
            If True, display grid map on screen with pylab.show()
            and produce an output PDF file. Default is: False.
            
        Returns
        -------
        out : HDF5 file
            File containing arrays of s and q for each of the sub-grids.
        out : pylab.show() and PDF file
            If plot=True, display grid map on screen and produce a PDF file.
            
        Examples
        --------
        >>> grs = GridSearch()
        >>> grs = gengridlibs('gridlib-0')
        >>> grs.gengridlibs('gridlib-1', srange=(0.8, 1.25), qrange=(0.001, 0.1), majgrid=(3, 3), mingrid=(6, 6), axis=[0.7, 1.4, 0.0006, 0.2], plot=True)
        """
    # set I/O shell display
    tit, tcol, tend = "\033[0m\033[3m", "\033[0m\033[35m", "\033[0m"

    # check whether the grid name does not contain extensions
    if '.hdf5' in gridsprefix:
        raise NameError("name should not contain extension")

    # create file names
    pdfname = gridsprefix + '.pdf'
    libname = gridsprefix + '.hdf5'

    # check weather grid already exists
    if os.path.isfile(libname):
        raise IOError("file '" + libname + "' already exists")

    # define sub-grids
    smin, smax = srange[0], srange[1]
    qmin, qmax = qrange[0], qrange[1]
    Ngs, Ngq = majgrid[0], majgrid[1]
    Ns, Nq = mingrid[0], mingrid[1]
    S = np.empty([Ngs, Ns], dtype=np.float_)
    Q = np.empty([Ngq, Nq], dtype=np.float_)
    fullS = np.geomspace(smin, smax, Ngs * Ns, endpoint=True)
    fullQ = np.geomspace(qmin, qmax, Ngq * Nq, endpoint=True)
    for i in range(Ngs):
        S[i,] = fullS[i * Ns:(i + 1) * Ns]
    for j in range(Ngq):
        Q[j,] = fullQ[j * Nq:(j + 1) * Nq]

    # verbose
    printi(tcol + "Create grid " + tit + "'" + libname + "'" + tcol + " (view configuration:" + tit + "'" + pdfname + "'" + tcol + ")" + tend)
    printd(tit + "  (" +  str(smin) + " ≤ s ≤ " + str(smax) + ", " + str(qmin) + " ≤ q ≤ " + str(qmax) + ", " + str(Ngs) + " x " + str(Ngq) + " sub-grids, each of size " + str(Ns) + " x " + str(Nq) + ")" + tend)

    # create individual grids and store in HDF5 file
    grids = list()
    with h5py.File(libname, 'w') as gridlib:
        gridlib.attrs['Ngs'] = np.shape(S)[0]
        gridlib.attrs['Ngq'] = np.shape(Q)[0]
        
        gridlib.attrs['nmc'] = nmc
        gridlib.attrs['pcaus'] = pcaus
        
        for j in range(np.shape(Q)[0]):
            for i in range(np.shape(S)[0]):
                grids.append((S[i,], Q[j,]))
                gridlibk = gridlib.create_group(u'' + str(i) + ' ' + str(j))
                gridlibk.create_dataset('s', data=S[i,])
                gridlibk.create_dataset('q', data=Q[j,])
        gridlib.flush()
        gridlib.close()

    # plot template grid
    plt.rc('font', size=14)
    plt.close('all')
    fig, MAP = plt.subplots(1, figsize=(8,6))
    MAP.set_xscale('log')
    MAP.set_yscale('log')
    MAP.set_xlim([axis[0], axis[1]])
    MAP.set_ylim([axis[2], axis[3]])
    plt.subplots_adjust(left=0.16, bottom=0.13, right=0.94, top=0.94, wspace=None, hspace=None)
    MAP.set_title(r'Binary-lens search map template')
    MAP.set_xlabel(r'$s$')
    MAP.set_ylabel(r'$q$')
    for k in range(len(grids)):
        SP, QP = np.meshgrid(grids[k][0], grids[k][1])
        MAP.scatter(SP, QP, marker='*', s=12)
        plt.text(grids[k][0][0], grids[k][1][0], str(k), fontsize=15)
    plt.xticks(np.array([0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5]), np.array(['0.2', '0.3', '0.5', '0.7', '1', '2', '3', '4', '5']))
    plt.savefig(pdfname)

def compute_subgrids(gridsprefix, gridlist, nprocs=1, f_rcroi=2.):
    """Create magnification curve sub-grids (HDF5 files)"""
    # set I/O shell display
    tcol, tit, tend = "\033[0m\033[31m", "\033[0m\033[3m", "\033[0m"
    
    # check whether input names does not contain extensions
    if '.hdf5' in gridsprefix:
        raise NameError("grid prefix should not contain .hdf5 extension")
    
    # create name
    grid = gridsprefix + '.hdf5'

    # check weather library exists
    if not os.path.isfile(grid):
        raise IOError("file '" + grid + "' is missing")

    # verbose
    printd(tcol + "Grid " + tit + "'" + grid + "'" + tcol + " chosen" + tend)

    # mutliprocessing: create grid list names
    listmclibs, listgrids = list(), list()
    with h5py.File(grid, 'r') as fgrid:
        Ngs = fgrid.attrs['Ngs']
        Ngq = fgrid.attrs['Ngq']
        
        nmc = fgrid.attrs['nmc']
        pcaus = fgrid.attrs['pcaus']
        
        k = 0
        for j in range(Ngq):
            for i in range(Ngs):
                if k in gridlist:
                    # list of mc libraries to process
                    mclibk = gridsprefix + '_' + str(k) + '.hdf5'
                    
#                    # if file exist, abort --> non on complete maintenant
#                    if os.path.isfile(mclibk):
#                        raise IOError("file '" + mclibk + "' already exists")

                    # add mc library to to-process list
                    listmclibs.append(mclibk)
                    
                    # list of corresponding s,q values
                    gridi = fgrid[str(i) + ' ' + str(j)]
                    listgrids.append((gridi['s'][:], gridi['q'][:]))
                k += 1
        fgrid.flush()
        fgrid.close()

    # mutliprocessing: create arguments of _process_grids, and create workers pool
    printi(tcol + "Starting manager with PID " + tit + str(os.getpid()) + tcol + " running " + tit + str(nprocs) + tcol + " process(es)" + tend)
    listargs = zip(listmclibs, listgrids, repeat(nmc), repeat(pcaus), repeat(f_rcroi))
    pool = Pool(processes=nprocs)
    pool.imap_unordered(_process_grids, listargs)
    
    # collect results
    pool.close()
    pool.join()

def _process_grids((mclib, grid, nmc, pcaus, f_rcroi)):
    """Process of compute_subgrids"""
    # set I/O shell display
    tun, tcol, tit, tend = "\033[0m\033[1;31m", "\033[0m\033[31m", "\033[0m\033[3m", "\033[0m"
    
    # verbose
    printi(tcol + "Launching " + tit + "'" + mclib + "'" + tcol + " grid with PID " + tit + str(os.getpid()) + tend)
    
    # create mc of current sub-grid
    mc = MagnificationCurve()
    params = dict()
    k = 0
    for params['s'] in grid[0]:
        for params['q'] in grid[1]:
            
            # get reference parameters of mc
            mc.create({'s': params['s'], 'q': params['q']}, calcmc=False)
            
            # compute mc grid
            grpname = str(k)
            for id in range(nmc):
                
                mcid = grpname + '/' + str(id)
                
                # check if dataset exists
                go = True
                if os.path.isfile(mclib):
                    
                    with h5py.File(mclib, 'r') as fmclib:
                        go = mcid not in fmclib
                        fmclib.flush()
                        fmclib.close()
            
                if go:

                    # generate random central/secondary trajectories
                    croi = np.random.choice(['central', 'secondary'], p=[pcaus, 1. - pcaus])
                    if mc.content['topo'] == 'interm':
                        cx, cy, r = mc.content['croi']['resonant']
                    if mc.content['topo'] == 'close':
                        if croi == 'secondary':
                            cx, cy, r = mc.content['croi']['secondary_up']
                        else:
                            cx, cy, r = mc.content['croi']['central']
                    if mc.content['topo'] == 'wide':
                        cx, cy, r = mc.content['croi'][croi]
                    
                    # generate rho and alpha
                    params['rho'] = np.power(10., np.random.uniform(-3.5, -1.5))
                    params['alpha'] = np.random.uniform(0., np.pi / 2.)
                    
                    # generate u0
                    #   u0c: trajectory through selected croi center
                    u0c = - cx * np.sin(params['alpha']) + cy * np.cos(params['alpha'])
                    #   uc: local centered on selected croi
                    ucm = f_rcroi * r
                    uc = np.random.uniform(-ucm, ucm)
                    params['u0'] = uc + u0c

                    # create mc
                    mc.create(params)
               
                    # write metadata and mc
                    attrs = {'Ns': len(grid[0]), 'Nq': len(grid[1]), grpname + '/s': params['s'], grpname + '/q': params['q'], mcid + '/refcroi': croi}
            
                    mc.write(mclib, mcid, attrs=attrs)
                    
                else:
                    printi(tcol + "Magnification curve '" + tit + mcid + "'" + tcol + "already exists : skipping" + tend)
            k += 1

    # verbose
    printi(tun + "Magnification curve grid " + tit + "'" + mclib + "'" + tun + " complete" + tend)

def fit_subgrids(gridsprefix, fitsprefix, datasets, gridlist, init=None, trange=None, nprocs=1, overwrite=False):
    """Fit light curve on magnification curves grids
        
        IDEE : on pourra utiliser un random des 1000 pour faire
        des maps a faible resolution !
        """
    # set I/O shell display
    tcol, tit, tend = "\033[0m\033[31m", "\033[0m\033[3m", "\033[0m"
    
    # check whether input names does not contain extensions
    if '.hdf5' in gridsprefix:
        raise NameError("grid prefix should not contain .hdf5 extension")
    if '.hdf5' in fitsprefix:
        raise NameError("fit prefix should not contain .hdf5 extension")

    # delete existing HDF5 files in fits/
    if overwrite:
        printd(tcol + "Removing previous HDF5 files from fits/" + tend)
        proc = subprocess.Popen('rm -rf  ' + fitsprefix + '*.hdf5', shell=True, executable='/bin/bash')
        proc.wait()

    # mutliprocessing: create grid list names
    listmclibs, listlclibs = list(), list()
    for gridi in gridlist:
        
        mclib = gridsprefix + '_' + str(gridi) + '.hdf5'
        lclib = fitsprefix + '_' + str(gridi) + '.hdf5'
        listmclibs.append(mclib)
        listlclibs.append(lclib)
    
    # mutliprocessing: create arguments of _process_fits, and create workers pool
    printi(tcol + "Starting manager with PID " + tit + str(os.getpid()) + tcol + " running " + tit + str(nprocs) + tcol + " process(es)" + tend)
    listargs = zip(listmclibs, listlclibs, repeat(datasets), repeat(trange), repeat(init))
    pool = Pool(processes=nprocs)
    pool.imap_unordered(_process_fits, listargs)
    
    # collect results
    pool.close()
    pool.join()

def _process_fits((mclib, lclib, datasets, trange, init)):
    """Process of fit_subgrids"""
    
    # set I/O shell display
    tfil, tun, tcol, tit, tend = "\033[0m\033[1;35m", "\033[0m\033[1;31m", "\033[0m\033[31m", "\033[0m\033[3m", "\033[0m"
    
    # check that mclib(.hdf5) exists
    if not os.path.isfile(mclib):
        raise IOError("file '" + mclib + "' is missing")

    # verbose
    printi(tcol + "Launching " + tit + "'" + mclib + "'" + tcol + " grid with PID " + tit + str(os.getpid()) + tend)
    
    with h5py.File(mclib, 'r') as fmclib:
        # NEW read datasets and time range
        mc = MagnificationCurve()
        lc = LightCurve(datasets, trange=trange)

        # global subgrid attributes
        Ns = fmclib.attrs['Ns']
        Nq = fmclib.attrs['Nq']

        # prepare grid
        grids = np.empty(Ns * Nq, dtype=np.float_)
        gridq = np.empty(Ns * Nq, dtype=np.float_)
        gridu0 = np.empty(Ns * Nq, dtype=np.float_)
        gridalpha = np.empty(Ns * Nq, dtype=np.float_)
        gridtE = np.empty(Ns * Nq, dtype=np.float_)
        gridt0 = np.empty(Ns * Nq, dtype=np.float_)
        gridrho = np.empty(Ns * Nq, dtype=np.float_)
        gridchidof = np.empty(Ns * Nq, dtype=np.float_)
        gridchi = np.empty(Ns * Nq, dtype=np.float_)
        bestmc = np.empty(Ns * Nq, dtype=np.dtype('a128'))
        for nsq in range(len(fmclib.keys())):
            sqlib = fmclib[str(nsq)]
            grids[nsq] = sqlib.attrs['s']
            gridq[nsq] = sqlib.attrs['q']
            fits = list()
            usefit = 0
            for id in sqlib:
                # read mc
                f_u0, f_alpha, f_tE, f_t0, f_rho, f_chidof, f_chi = [],[],[],[],[],[],[]
                mc.read(mclib, str(nsq) + '/' + id)

                # fit only if ∆mag(th) > ∆mag(exp)
                dmag = 2.5 * np.log10(fmclib[str(nsq) + '/' + id].attrs['mumax'])
                
                if dmag < lc.content['dmag']:
                    printi(tfil + "Model delta(mag) too low : skipping" + tend)
                    printd(tit + "  (delta(mag_th) = " + str(dmag) + " < " + str(lc.content['dmag']) + tend)
                else:
                    usefit += 1
                    
                    # read reference croi in mc
                    croi = fmclib[str(nsq) + '/' + id].attrs['refcroi']

                    # fit for 0 < alpha < π/2
                    lc.fit(mc, croi, '+', init=init)
                    f_u0.append(lc.params['u0'])
                    f_alpha.append(lc.params['alpha'])
                    f_tE.append(lc.params['tE'])
                    f_t0.append(lc.params['t0'])
                    f_rho.append(lc.params['rho'])
                    f_chidof.append(lc.content['chi2'][0] / lc.content['chi2'][1])
                    f_chi.append(lc.content['chi2'][0])

                    # fit for π < alpha < 3π/2)
                    lc.fit(mc, croi, '-', init=init)
                    f_u0.append(lc.params['u0'])
                    f_alpha.append(lc.params['alpha'])
                    f_tE.append(lc.params['tE'])
                    f_t0.append(lc.params['t0'])
                    f_rho.append(lc.params['rho'])
                    f_chidof.append(lc.content['chi2'][0] / lc.content['chi2'][1])
                    f_chi.append(lc.content['chi2'][0])

                    # add fit to list if chi2 is not inf
                    if not np.all(np.isinf(f_chidof)):
                        arg = np.argmin(f_chidof)
                        u0 = f_u0[arg]
                        alpha = f_alpha[arg]
                        tE = f_tE[arg]
                        t0 = f_t0[arg]
                        rho = f_rho[arg]
                        chidof = f_chidof[arg]
                        chi = f_chi[arg]
                        fits.append([id, u0, alpha, tE, t0, rho, chidof, chi])

            # verbose
            printd(tcol + "Percentage of useful magnification curves is about " + tit + "{0:.0f}".format(100. * float(usefit) / float(id)) + "%" + tend)
            
            if fits:
                # sort fits by increasing chi2 and get parameters
                fits = np.array(fits)
                arg = np.argsort(np.array(fits[:, 6], dtype=np.float_))
                mcs = np.array(fits[arg, 0], dtype=np.int_)
                u0 = np.array(fits[arg, 1], dtype=np.float_)
                alpha = np.array(fits[arg, 2], dtype=np.float_)
                tE = np.array(fits[arg, 3], dtype=np.float_)
                t0 = np.array(fits[arg, 4], dtype=np.float_)
                rho = np.array(fits[arg, 5], dtype=np.float_)
                chidof = np.array(fits[arg, 6], dtype=np.float_)
                chi = np.array(fits[arg, 7], dtype=np.float_)
                
                # save best-fit parameters and chi2/dof
                gridu0[nsq] = u0[0]
                gridalpha[nsq] = alpha[0]
                gridtE[nsq] = tE[0]
                gridt0[nsq] = t0[0]
                gridrho[nsq] = rho[0]
                gridchidof[nsq] = chidof[0]
                gridchi[nsq] = chi[0]
                bestmc[nsq] = str(nsq) + '/' + str(mcs[0])

                # verbose
                printi(tcol + "Best-fit model at grid point " + tit + "'" + str(nsq) + "'" + tcol + " in file " + tit + mclib + tcol + " is " + tit + "'" + str(mcs[0]) + "'" + tcol + " with " + tit + "chi2/dof={:.3e}".format(chidof[0]) + tend)
            else:
                gridchidof[nsq] = np.inf
                gridchi[nsq] = np.inf

        # save log(X^2) map in HDF5 file: overwrite existing file
        with h5py.File(lclib, 'w') as fitres:
            gS = np.unique(grids)
            gQ = np.unique(gridq)
            gs, gq = np.meshgrid(gS, gQ)
            
            fitres.create_dataset('s', data=gs)
            fitres.create_dataset('q', data=gq)
            fitres.create_dataset('u0', data=gridu0.reshape(Ns, Nq).T)
            fitres.create_dataset('alpha', data=gridalpha.reshape(Ns, Nq).T)
            fitres.create_dataset('tE', data=gridtE.reshape(Ns, Nq).T)
            fitres.create_dataset('t0', data=gridt0.reshape(Ns, Nq).T)
            fitres.create_dataset('rho', data=gridrho.reshape(Ns, Nq).T)
            fitres.create_dataset('chidof', data=gridchidof.reshape(Ns, Nq).T)
            fitres.create_dataset('chi', data=gridchi.reshape(Ns, Nq).T)

            fitres.flush()
            fitres.close()
        fmclib.flush()
        fmclib.close()

    # verbose
    printi(tun + "Light curve grid " + tit + "'" + lclib + "'" + tun + " complete" + tend)
