# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

# version: 8.9.21, 27.5.21, 11.3.21

import numpy as np
import matplotlib.pylab as plt
from utils import verbosity, printi, printd, printw

### SET verbosity level
#verbosity('DEBUG')
#verbosity('NONE')
verbosity('INFO')

# set I/O shell display
tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"

I = np.complex(0., 1.)

# data in (W, N) coordinates
#hjd, uW, uN = np.loadtxt('Gaia19bld_WN_src_traj.dat', unpack=True) # 5.12.2020
hjd, uW, uN = np.loadtxt('Gaia19bld_WN_src_traj_NEW.dat', unpack=True)
hjd1 = hjd - 2458681.

# convert to lens-source and W->E
uN1, uE1 = -uN, uW

# PIONIER epochs (JD-2458681)
t12a = -3.523433043155819
t12b =-3.5002864212729037
t19a = 3.465681141242385
t19b = 3.4803526680916548
t21 = 5.493884213734418

# t0_par
t0par = -1.

# limit main array
tm = 150
arg = np.logical_and(-tm <= hjd1, hjd1 <= tm)
hjd1 = hjd1[arg]
uE1 = uE1[arg]
uN1 = uN1[arg]

# limit array larger than PIONIER (±1.5)
arg = np.logical_and(t12a-1.5 <= hjd1, hjd1 <= t21+1.5)
hjd2 = hjd1[arg]
uE2 = uE1[arg]
uN2 = uN1[arg]

# fit parabola
coef = np.polyfit(hjd2, uE2, 2)
fpE = np.poly1d(coef)
coef = np.polyfit(hjd2, uN2, 2)
fpN = np.poly1d(coef)

# velocities in the (N, W) frame
v = fpE.deriv() + I * fpN.deriv()

# limit array PIONIER exact date interval
arg = np.logical_and(t12a <= hjd2, hjd2 <= t21)
hjd = hjd2[arg]
uE = uE2[arg]
uN = uN2[arg]

# plot figure to show alpha variation
plt.close('all')
plt.figure(figsize=(13, 4))
plt.rc('font', size=15)

# min, max values of to
tmin = t12a - 1.
tmax = t21 + 1.

ax = plt.subplot(1, 2, 1)
ax.text(-0.13, 0.97, 'a', weight='bold', transform=ax.transAxes)
ax.plot(uE1, uN1, color='k', lw=0.7)
ax.plot(uE, uN, color='k', lw=4)
ax.set_xlabel(r'$x$ $[\theta_{\rm E}]\quad$ West $\longrightarrow$')
ax.set_ylabel(r'$y$ $[\theta_{\rm E}]\quad$ North $\longrightarrow$')
ax.set_xlim([np.amax(uE1), np.amin(uE1)])
ax.set_aspect('equal')
ax.set_xticks([-1, -0.5, 0., 0.5, 1])
ax.set_yticks([-0.5, 0., 0.5])

ax = plt.subplot(1, 2, 2)
ax.text(-0.13, 0.97, 'b', weight='bold', transform=ax.transAxes)
ax.plot(hjd2, np.angle(-np.conj(v(hjd2)), deg=True) % 360, color='k', lw=0.7)
ax.plot(hjd, np.angle(-np.conj(v(hjd)), deg=True) % 360, color='k', lw=4)
ax.set_xlabel('JD-2,458,681')
ax.set_ylabel(r'$\alpha^\prime$ (°)')
ax.set_xlim([tmin, tmax])
ax.set_ylim([150.6, 154.6])

## SET best-fit values and error (en fait median et 1-sigma)
alphac = 152.6430
alpham = alphac - 0.9494
alphap = alphac + 0.9328
np.array([alpham, alphap])

# central value and uncertainty on alpha^prime
ax.plot([tmin, tmax], [alphac, alphac], c='k', lw=1., ls='--')
ax.fill_between(np.array([t12a, t21]), np.array([alpham, alpham]), np.array([alphap, alphap]), fc='silver', linewidth=0.3, edgecolor='darkgray')

# t0_par and text
ax.plot([t0par, t0par], [150, 155], c='k', lw=1., ls='--')
ax.text(t0par, 150.3, r'$t_{0, {\rm par}}$', size=12)

plt.tight_layout()

plt.savefig('Cassan_EDFig7.eps') # fig_alpha.pdf
