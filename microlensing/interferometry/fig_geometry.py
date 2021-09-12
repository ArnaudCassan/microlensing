# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 5/12/2020

import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Arc
from utils import verbosity, printi, printd, printw
from ESPL import newz
from fig_degen import fig_degen0, fig_degen2, fig_degen3
from fig_paras import fig_paras

def fig_geometry(figname=None):
    """Combined figure to show fitting parameters.
    """
    # combined plot design
    plt.close('all')
    plt.figure(figsize=(8, 7.5))
    plt.subplots_adjust(top=0.99, bottom=0.06, left=0.09, right=0.99, wspace=0.3)
    
    xylim = [-1.3, 1.3]
    xlabel = r'$x_{\rm E}$ $[\theta_{\rm E}]\quad$ West $\longrightarrow$'
    ylabel = r'$y_{\rm E}$ $[\theta_{\rm E}]\quad$ North $\longrightarrow$'
    
    numfig = (-0.13, 0.97)
    rev = ['', r'$1.0$', r'$0.5$', r'$0.0$', r'$-0.5$', r'$-1.0$']
    
    # parameters
    ax = plt.subplot(2, 2, 1)
    ax.text(*numfig, 'a', weight='bold', transform=ax.transAxes)
    ax.set_ylim(xylim)
    ax.set_xlim(xylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels(rev)
    fig_paras(ax=ax)
    
    # non-distinguishable arcs
    ax = plt.subplot(2, 2, 2)
    ax.text(*numfig, 'b', weight='bold', transform=ax.transAxes)
    ax.set_ylim(xylim)
    ax.set_xlim(xylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels(rev)
    fig_degen0(ax=ax)
    
    # 2 epochs
    ax = plt.subplot(2, 2, 3)
    ax.text(*numfig, 'c', weight='bold', transform=ax.transAxes)
    ax.set_ylim(xylim)
    ax.set_xlim(xylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels(rev)
    fig_degen2(ax=ax)
    
    # 3 epochs
    ax = plt.subplot(2, 2, 4)
    ax.text(*numfig, 'd', weight='bold', transform=ax.transAxes)
    ax.set_ylim(xylim)
    ax.set_xlim(xylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels(rev)
    fig_degen3(ax=ax)

    # save figure
    if figname is not None:
        plt.savefig(figname)

if __name__ == '__main__':

    ## SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')
   
    fig_geometry(figname='Cassan_EDFig4.eps') # fig_geometry.pdf

