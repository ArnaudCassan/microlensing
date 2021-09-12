# Database and three figures by Clément Ranc.
# Combined figures layout adpated by Arnaud Cassan.

# version: 8.9.21, 14.6.21, 31.5.21, 28.5.21, 26/01/2021

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import verbosity, printi, printd, printw

def fig_piE(ev_nasa, ev_zhu, figname=None, axc=None):
    """Plot relative uncertainties on pi_E.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

    # Load data from NASA Exoplanet Archive
    data = pd.read_fwf(ev_nasa, index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Load data from Zhu+17
    zhu = pd.read_table(ev_zhu, sep='\s+', header=0, skiprows=0)

    # Calculations for Zhu+17
    zhu['piE'] = np.sqrt(np.sum(zhu[['piee', 'pien']].values**2, axis=1))
    zhu['piEerr'] = np.sqrt(np.sum(zhu[['err_pien', 'err_piee']].values**2, axis=1))
    zhu['piErm'] = zhu['piEerr'] / zhu['piE']
    zhu_events = np.unique(zhu['name'])

    # Identify which event to plot
    data['flag_plot'] = False
    for i in range(len(data)):
        kwd = data['plot'].values[i].split('+')
        if 'piE' in kwd:
            data.loc[i, 'flag_plot'] = True

    # single figure design
    if axc is None:
        fig, axc = plt.subplots(1, 1, figsize=[6, 4])
        axc.set_xlim(3., 300.)
        axc.set_xlabel(r'$t_{\rm E}$ [days]')
    axc.set_ylim(0.007, 1.5)
    axc.set_xscale('log')
    axc.set_yscale('log')
    axc.set_ylabel(r'$\pi_{\rm E}$ relative uncertainty')

    # --- Data ---
    mask = data['flag_plot'] & (data['method'] == 'highres')
    x = data[mask]['t_E']
    y = data[mask]['pi_E_mean_err']
    axc.scatter(x, y, 19, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3, label='High-resolution')

    mask = data['flag_plot'] & (data['method'] == 'spacehighres')
    x = data[mask]['t_E']
    y = data[mask]['pi_E_mean_err']
    #axc.scatter(x, y, s=19, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3)
    axc.scatter(x, y, s=30, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.2)
    axc.scatter(x, y, s=10, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3)

    mask = data['flag_plot'] & (data['method'] == 'standard')
    x = data[mask]['t_E']
    y = data[mask]['pi_E_mean_err']
    axc.scatter(x, y, s=16, marker='s', c='k', lw=0, zorder=0.1, label='Ground')

    mask = data['flag_plot'] & (data['method'] == 'spacepie')
    x = data[mask]['t_E']
    y = data[mask]['pi_E_mean_err']
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.2, label=r'Ground+Space')

    mask = data['flag_plot']
    for i in range(mask.sum()):
        x = [data[mask]['t_E'].values[i], data[mask]['t_E'].values[i]]
        y = [data[mask]['pi_E_lower_err'].values[i], data[mask]['pi_E_upper_err'].values[i]]
        if not y[0] == y[1]:
            axc.plot(x, y, c='k', dashes=(0.8,0.8), lw=0.6, zorder=0)

    # Zhu and Spitzer
    for i in range(len(zhu_events)):
        name = zhu_events[i]
        maskz = zhu['name'] == name
        x = zhu.loc[maskz, 'te']
        y = zhu.loc[maskz, 'piErm']
        if i==0: axc.plot(x, y, c='k', lw=0.6, zorder=0, label='Degeneracy')
        else: axc.plot(x, y, c='k', lw=0.6, zorder=0)

    x = zhu['te']
    y = zhu['piErm']
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.31)

    # Add Gaia19bld
    x = [106.65]
#    y = [0.0021 / 0.082335]
    y = [0.0020640705606162955 / 0.08187744162231324]
    printi(tcol+"Gaia19bld tE, piE : "+tend+f"{x}, {y}")
    axc.scatter(x, y, s=40, marker='*', fc='b', ec='b', lw=0.8)
    axc.annotate(r'Gaia19bld',
        xy=(x[0], y[0]), xycoords='data', c='b',
        xytext=(30, -8), textcoords='offset points',
        size=10, va='center', ha='center')

    # Add Kojima-1
    x = [27.89]
    #y = [0.063 / 0.495] # ancien
    y = [0.060 / 0.469] # Zang luminous model
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=3.9)
    axc.annotate(r'Kojima-1',
        xy=(x[0], y[0]), xycoords='data', c='k', rotation=0,
        xytext=(-18, -24), textcoords='offset points',
        size=8, va='top', ha='center',
        bbox=dict(fc='none', ec='none', pad=0),
        arrowprops=dict(arrowstyle='-|>', fc='k', lw=0.4, shrinkA=1, shrinkB=3,
        connectionstyle='arc3, rad=-0', relpos=(0.8, 1)))

    # legend
    axc.legend(loc='upper left', ncol=1, fontsize='x-small')
        
    if figname is not None:
        fig.savefig(figname, transparent=False, bbox_inches='tight', dpi=300, pad_inches=0.01)

def fig_thE(ev_nasa, figname=None, axc=None):
    """Plot relative uncertainties on theta_E.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

    # Load data from NASA Exoplanet Archive
    data = pd.read_fwf(ev_nasa, index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Identify which event to plot
    data['flag_plot'] = False
    for i in range(len(data)):
        kwd = data['plot'].values[i].split('+')
        if 'thE' in kwd:
            data.loc[i, 'flag_plot'] = True

    # single figure design
    if axc is None:
        fig, axc = plt.subplots(1, 1, figsize=[6, 4])
        axc.set_xlim(3., 300.)
        axc.set_xlabel(r'$t_{\rm E}$ [days]')
    axc.set_ylim(0.003, 0.7)
    axc.set_xscale('log')
    axc.set_yscale('log')
    axc.set_ylabel(r'$\theta_{\rm E}$ relative uncertainty')

    # --- Data ---
    mask = data['flag_plot'] & (data['method'] == 'highres')
    x = data[mask]['t_E']
    y = data[mask]['theta_E_mean_err']
    axc.scatter(x, y, 19, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3, label='High-resolution')

    mask = data['flag_plot'] & (data['method'] == 'spacehighres')
    x = data[mask]['t_E']
    y = data[mask]['theta_E_mean_err']
    #axc.scatter(x, y, s=19, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3)
    axc.scatter(x, y, s=30, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.2)
    axc.scatter(x, y, s=10, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3)


    mask = data['flag_plot'] & (data['method'] == 'standard')
    x = data[mask]['t_E']
    y = data[mask]['theta_E_mean_err']
    axc.scatter(x, y, s=16, marker='s', c='k', lw=0, zorder=0.1, label='Ground')

    mask = data['flag_plot'] & (data['method'] == 'spacepie')
    x = data[mask]['t_E']
    y = data[mask]['theta_E_mean_err']
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.2, label=r'Space parallax')

    mask = data['flag_plot']
    for i in range(mask.sum()):
        x = [data[mask]['t_E'].values[i], data[mask]['t_E'].values[i]]
        y = [data[mask]['theta_E_lower_err'].values[i], data[mask]['theta_E_upper_err'].values[i]]
        if not y[0] == y[1]:
            axc.plot(x, y, c='k', dashes=(0.8,0.8), lw=0.6, zorder=0)

    # Add Gaia19bld
    x = [106.65]
    y = [0.0038836 / 0.765008]
    printi(tcol+"Gaia19bld tE, thE : "+tend+f"{x}, {y}")
    axc.scatter(x, y, s=40, marker='*', fc='b', ec='b', lw=0.8)
    axc.annotate(r'Gaia19bld',
        xy=(x[0], y[0]), xycoords='data', c='b',
        xytext=(30, -8), textcoords='offset points',
        size=10, va='center', ha='center')

    # Add Kojima-1
    x = [27.89]
    #y = [0.03 / 1.87] # not luminous/luminous model
    y = [0.014 / 1.891] # Zang luminous model
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=3.9)
    axc.annotate(r'Kojima-1',
        xy=(x[0], y[0]), xycoords='data', c='k', rotation=0,
        xytext=(-14, 16), textcoords='offset points',
        size=8, va='bottom', ha='center',
        bbox=dict(fc='none', ec='none', pad=0),
        arrowprops=dict(arrowstyle='-|>', fc='k', lw=0.4, shrinkA=4, shrinkB=4,
        connectionstyle='arc3, rad=0', relpos=(0.6, 0)))

    if figname is not None:
        # legend
        axc.legend(loc='upper left', ncol=1, fontsize='x-small')
        fig.savefig(figname, transparent=False, bbox_inches='tight', dpi=300, pad_inches=0.01)

def fig_M(ev_nasa, figname=None, axc=None):
    """Plot relative uncertainties on lens mass M.
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

    # Load data from NASA Exoplanet Archive
    data = pd.read_fwf(ev_nasa, index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Identify which event to plot
    data['flag_plot'] = False
    for i in range(len(data)):
        kwd = data['plot'].values[i].split('+')
        if 'M' in kwd:
            data.loc[i, 'flag_plot'] = True

    # single figure design
    if axc is None:
        fig, axc = plt.subplots(1, 1, figsize=[6, 4])
        axc.set_xlim(3., 300.)
    axc.set_ylim(0.01, 4)
    axc.set_xscale('log')
    axc.set_yscale('log')
    axc.set_xlabel(r'$t_{\rm E}$ [days]')
    axc.set_ylabel(r'$M$ relative uncertainty')
    
    # --- Data ---
    mask = data['flag_plot'] & (data['method'] == 'highres')
    x = data[mask]['t_E']
    y = data[mask]['mass_mean_err']
    axc.scatter(x, y, 19, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3, label='High-resolution')

    mask = data['flag_plot'] & (data['method'] == 'spacehighres')
    x = data[mask]['t_E']
    y = data[mask]['mass_mean_err']
    #axc.scatter(x, y, s=19, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3)
    axc.scatter(x, y, s=30, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.2)
    axc.scatter(x, y, s=10, marker='d', fc='limegreen', ec='k', lw=0.4, zorder=0.3)


    mask = data['flag_plot'] & (data['method'] == 'standard')
    x = data[mask]['t_E']
    y = data[mask]['mass_mean_err']
    axc.scatter(x, y, s=16, marker='s', c='k', lw=0, zorder=0.1, label='Ground')

    mask = data['flag_plot'] & (data['method'] == 'spacepie')
    x = data[mask]['t_E']
    y = data[mask]['mass_mean_err']
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=0.2, label=r'Space parallax')

    mask = data['flag_plot']
    for i in range(mask.sum()):
        x = [data[mask]['t_E'].values[i], data[mask]['t_E'].values[i]]
        y = [data[mask]['mass_lower_err'].values[i], data[mask]['mass_upper_err'].values[i]]
        if not y[0] == y[1]:
            axc.plot(x, y, c='k', dashes=(0.8,0.8), lw=0.6, zorder=0)

    # Add Gaia19bld
    x = [106.65]
#    y = [0.029742 / 1.1406]
    y = [0.02954476924125032 / 1.1470886557408346]
    printi(tcol+"Gaia19bld tE, mass : "+tend+f"{x}, {y}")
    axc.scatter(x, y, s=40, marker='*', fc='b', ec='b', lw=0.8)
    axc.annotate(r'Gaia19bld',
        xy=(x[0], y[0]), xycoords='data', c='b',
        xytext=(30, -8), textcoords='offset points',
        size=10, va='center', ha='center')

    # Add Kojima-1
    x = [27.89]
    #y = [0.063 / 0.495] # ancien
    y = [0.032 / 0.527] # Zang luminous model
    axc.scatter(x, y, s=19, marker='o', fc='#ee342b', ec='k', lw=0.6, zorder=3.9)
    axc.annotate(r'Kojima-1',
        xy=(x[0], y[0]), xycoords='data', c='k', rotation=0,
        xytext=(-20, -16), textcoords='offset points',
        size=8, va='top', ha='center',
        bbox=dict(fc='none', ec='none', pad=0),
        arrowprops=dict(arrowstyle='-|>', fc='k', lw=0.4, shrinkA=1, shrinkB=3,
        connectionstyle='arc3, rad=-0', relpos=(0.8, 1)))

    if figname is not None:
        # legend
        axc.legend(loc='upper right', ncol=1, fontsize='x-small')
        fig.savefig(figname, transparent=False, bbox_inches='tight', dpi=300, pad_inches=0.01)

def fig_combined(ev_nasa, ev_zhu, figname=None):
    """Combined plots.
    """
    # combined plot design
    plt.close('all')
    plt.subplots(figsize=(5, 7.8))
    plt.subplots_adjust(top = 0.99, bottom=0.07, left=0.22, right=0.98, hspace=0)

    # fig ref pos
    xp, yp = -0.28, 0.96
    
    # piE
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_xlim(3., 300.)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.grid(ls='-', lw=0.3, which='both')
    ax1.text(xp, yp, 'a', weight='bold', transform=ax1.transAxes)
    fig_piE(ev_nasa, ev_zhu, axc=ax1)
    ax1.yaxis.set_ticklabels(['', '', '    1%', ' 10%', '100%'])
       
    # thetaE
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.grid(ls='-', lw=0.3, which='both')
    ax2.text(xp, yp, 'b', weight='bold', transform=ax2.transAxes)
    fig_thE(ev_nasa, axc=ax2)
    ax2.yaxis.set_ticklabels(['', '', '    1%', ' 10%'])

    # M
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.grid(ls='-', lw=0.3, which='both')
    ax3.text(xp, yp, 'c', weight='bold', transform=ax3.transAxes)
    fig_M(ev_nasa, axc=ax3)
    ax3.yaxis.set_ticklabels(['', '    1%', ' 10%', '100%'])
    ax3.xaxis.set_ticklabels(['', '', '10', '100'])
        
    plt.savefig(figname)
    
def planet_to_event_name(txt):
    res = txt.strip()
    res = res.replace('L b', '')
    res = res.replace('L c', '')
    res = res.replace('OGLE-2007-BLG-349L AB c', '   OGLE-2007-BLG-349')
    res = res.replace('OGLE-2016-BLG-0613L AB b', '   OGLE-2016-BLG-0613')
    res = res.replace('TCP J05074264+2447555 b', 'TCP J05074264+2447555')
    res = res.replace('OGLE-2013-BLG-0341L B b', 'OGLE-2013-BLG-0341')
    return res

def tableau_latex_piE(file_pie='pie.tex'):

    # Load data from NASA Exoplanet Archive
    data = pd.read_fwf(ev_nasa, index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Load data from Zhu+17
    zhu = pd.read_table(ev_zhu, sep='\s+', header=0, skiprows=0)
    zhu['piE'] = np.sqrt(np.sum(zhu[['piee', 'pien']].values**2, axis=1))
    zhu['piEerr'] = np.sqrt(np.sum(zhu[['err_pien', 'err_piee']].values**2, axis=1))
    zhu['pi_E_mean_err'] = zhu['piEerr'] / zhu['piE']
    zhu['planet_name'] = [f"OGLE-2015-BLG-{a:04d}" for a in zhu['name'].values]
    zhu['method'] = 'zhu'
    zhu['ads_reference'] = '2017AJ....154..210Z'
    zhu['t_E'] = zhu['te']
    zhu['plot'] = 'piE'

    # Concatenate the two databases
    data = pd.concat([data, zhu])
    data.reset_index(drop=True, inplace=True)

    # Identify which event to plot
    data['flag_plot'] = False
    for i in range(len(data)):
        kwd = data['plot'].values[i].split('+')
        if 'piE' in kwd:
            data.loc[i, 'flag_plot'] = True

    # Formatting for latex
    data['planet_name'] = [f"{planet_to_event_name(a):>12s}" for a in data['planet_name'].values]
    data['method'] = [f"{a:<24s}" for a in data['method'].values]

    col = ['planet_name', 'method', 'ads_reference', 'pi_E_mean_err']
    data.drop_duplicates(subset=col, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Prepare most of the table    
    mask = data['flag_plot']
    header = ['Event', 'Method', 'ADS Reference', 'Relative uncertainty']
    formatters = dict({
        'planet_name': "{:14s}".format, 
        'method': "{:s}".format, 
        'ads_reference': "{:19s}".format, 
        'pi_E_mean_err': "{:.4f}".format,
        })
    tab_tex = data[mask].sort_values('ads_reference').to_latex(columns=col, header=header, 
        index=False, formatters=formatters)

    # Correct some labels and text
    tab_tex = tab_tex.replace('Relative uncertainty', r'$\sigma(\pi_E)/\pi_E$')
    tab_tex = tab_tex.replace('NaN', '\dots')
    tab_tex = tab_tex.replace('spacepie                ', '{:24s}'.format('space'))
    tab_tex = tab_tex.replace('zhu                     ', '{:24s}'.format('space'))
    tab_tex = tab_tex.replace('spacehighres            ', '{:24s}'.format('space \& high resolution'))
    tab_tex = tab_tex.replace('highres                 ', '{:24s}'.format('high resolution'))
    tab_tex = tab_tex.replace('standard                ', '{:24s}'.format('annual'))

    # Save in a file
    file = open(file_pie, 'w')
    file.write(tab_tex)
    file.close()

def tableau_latex_thE(file_pie='thE.tex'):

    # Load data from NASA Exoplanet Archive
    data = pd.read_fwf(ev_nasa, index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Identify which event to plot
    data['flag_plot'] = False
    for i in range(len(data)):
        kwd = data['plot'].values[i].split('+')
        if 'thE' in kwd:
            data.loc[i, 'flag_plot'] = True

    # Formatting for latex
    data['planet_name'] = [f"{planet_to_event_name(a):>22s}" for a in data['planet_name'].values]
    data['method'] = [f"{a:<24s}" for a in data['method'].values]

    col = ['planet_name', 'method', 'ads_reference', 'theta_E_mean_err']
    data.drop_duplicates(subset=col, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Prepare most of the table    
    mask = data['flag_plot']
    header = ['Event', 'Method', 'ADS Reference', 'Relative uncertainty']
    formatters = dict({
        'planet_name': "{:14s}".format, 
        'method': "{:s}".format, 
        'ads_reference': "{:19s}".format, 
        'theta_E_mean_err': "{:.4f}".format,
        })
    tab_tex = data[mask].sort_values('ads_reference').to_latex(columns=col, header=header, 
        index=False, formatters=formatters)

    # Correct some labels and text
    tab_tex = tab_tex.replace('Relative uncertainty', r'$\sigma(\theta_E)/\theta_E$')
    tab_tex = tab_tex.replace('NaN', '\dots')
    tab_tex = tab_tex.replace('spacepie                ', '{:24s}'.format('space'))
    tab_tex = tab_tex.replace('spacehighres            ', '{:24s}'.format('space \& high resolution'))
    tab_tex = tab_tex.replace('highres                 ', '{:24s}'.format('high resolution'))
    tab_tex = tab_tex.replace('standard                ', '{:24s}'.format('annual'))

    # Save in a file
    file = open(file_pie, 'w')
    file.write(tab_tex)
    file.close()

def tableau_latex_M(file_pie='mass.tex'):

    # Load data from NASA Exoplanet Archive
    data = pd.read_fwf(ev_nasa, index_col=0)
    data.reset_index(drop=True, inplace=True)

    # Identify which event to plot
    data['flag_plot'] = False
    for i in range(len(data)):
        kwd = data['plot'].values[i].split('+')
        if 'M' in kwd:
            data.loc[i, 'flag_plot'] = True

    # Formatting for latex
    data['planet_name'] = [f"{planet_to_event_name(a):>22s}" for a in data['planet_name'].values]
    data['method'] = [f"{a:<24s}" for a in data['method'].values]

    col = ['planet_name', 'method', 'ads_reference', 'mass_mean_err']
    data.drop_duplicates(subset=col, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Prepare most of the table    
    mask = data['flag_plot']
    header = ['Event', 'Method', 'ADS Reference', 'Relative uncertainty']
    formatters = dict({
        'planet_name': "{:14s}".format, 
        'method': "{:s}".format, 
        'ads_reference': "{:19s}".format, 
        'mass_mean_err': "{:.4f}".format,
        })
    tab_tex = data[mask].sort_values('ads_reference').to_latex(columns=col, header=header, 
        index=False, formatters=formatters)

    # Correct some labels and text
    tab_tex = tab_tex.replace('Relative uncertainty', r'$\sigma(M)/M$')
    tab_tex = tab_tex.replace('NaN', '\dots')
    tab_tex = tab_tex.replace('spacepie                ', '{:24s}'.format('space'))
    tab_tex = tab_tex.replace('spacehighres            ', '{:24s}'.format('space \& high resolution'))
    tab_tex = tab_tex.replace('highres                 ', '{:24s}'.format('high resolution'))
    tab_tex = tab_tex.replace('standard                ', '{:24s}'.format('annual'))

    # Save in a file
    file = open(file_pie, 'w')
    file.write(tab_tex)
    file.close()


if __name__ == "__main__":

    ### SET verbosity level
    #verbosity('DEBUG')
    #verbosity('NONE')
    verbosity('INFO')
    
    # databases
    ev_nasa = 'event_list.dat'
    ev_zhu = 'zhu17.dat'
    
    # plot individual figures
    fig_piE(ev_nasa, ev_zhu, figname='fig_events_piE.pdf')
    fig_thE(ev_nasa, figname='fig_events_thE.pdf')
    fig_M(ev_nasa, figname='fig_events_M.pdf')

    # adapt font size
    plt.rc('font', size=12)
    
    # plot combined figure
    fig_combined(ev_nasa, ev_zhu, figname='Cassan_Fig4.eps') # fig_MthEpiE.pdf

    # Write corresponding LaTeX tables
    tableau_latex_piE()
    tableau_latex_thE()
    tableau_latex_M()

"""
Details about the NASA Exoplanet Archive database modification
##############################################################

Update of the ADS references
============================

I have updated the following references.

Column 1: reference found in NASA Exoplanets Database
Column 2: newer reference in Astrophysics Data System
2018arXiv180210067S     2018AcA....68...43S
2018arXiv180304437C     2018AJ....155..261C
2018arXiv180206795S     2018ApJ...857L...8S
2018arXiv180304437C     2018AJ....155..261C
2018arXiv180210067S     2018AcA....68...43S
2017arXiv171009974R     2018AJ....155...40R
2018arXiv180509983J     2018AJ....156..208J
2018arXiv181010792Z     2018AJ....156..236Z
2018arXiv180206659N     2018MNRAS.476.2962N
2018arXiv180305095J     2018AJ....155..219J
2017arXiv170501058M     2017AJ....154..205M
2018arXiv181112505S     2019AJ....157..146S
2019arXiv190501239K     2019AJ....158..224K
2019arXiv190711536N     2019AJ....158..212N
2020arXiv200409067H     2020AJ....160...74H
2019arXiv191203822J     2020AJ....160..255J

Filtering
=========

Step 1 is an automatic identification of keywords in the abstract:
- if the word ['Hubble' or 'Subaru' or 'Keck'] in the abstract, then classified as 'highres' in the column 'method' of file event_list.dat,
- if the word 'spitzer' in the abstract, then the method is 'spacepie'. Else, classified as 'standard' in file event_list.dat.

Step 2 is a manual check of the events classified as 'highres' (high resolution). After checking, we exclude the following events from the 'high resolution' class:
- 2008ApJ...684..663B
- 2014ApJ...780..123S
- 2017AJ....154....3K

# Plot 1: microlens parallax as a function of Einstein timescale

i) The Einstein timescale, tE, used is the value downloaded from the NASA Exoplanet Archive.

ii) From the NASA Exoplanet Archive, I select only the events that have both:
- a value of microlens parallax, piE, and
- a value of the error on piE.
These events are indicated in the column plot by 'piE'.

iii) We plot:
- the data from ii),
- the data from Zhu+2017 (file zhu17.dat).

# Plot 2: Angular Einstein radius as a function of Einstein timescale

i) The Einstein timescale, tE, used is the value downloaded from the NASA Exoplanet Archive.

ii) From the NASA Exoplanet Archive, I select only the events that have both:
- a value of angular Einstein radius, thE, and
- a value of the error on thE.
These events are indicated in the column plot by 'thE'.

iii) We plot:
- the data from ii).

# Plot 3: Lens mass as a function of Einstein timescale

i) The Einstein timescale, tE, used is the value downloaded from the NASA Exoplanet Archive.

ii) From the NASA Exoplanet Archive, I select only the events that have both:
- a value of lens mass, M, and
- a value of the error on M.
These events are indicated in the column plot by 'M'.

iii) We plot:
- the data from ii).
"""
