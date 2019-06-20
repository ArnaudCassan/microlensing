microlensing
======

<a href="https://travis-ci.org/ArnaudCassan/microlensing"><img src="https://travis-ci.org/ArnaudCassan/microlensing.svg?branch=master"></a>
<a href="http://ascl.net/1704.014"><img src="https://img.shields.io/badge/ascl-1704.014-blue.svg?colorB=262255" alt="ascl:1704.014" /></a>

Goals
-----

<b>microlensing</b> is a Python modelling package for gravitational microlensing

Getting Started
---------------

### Installation

To install the current development version of the <b>microlensing</b> package from source: 

```
$ git clone https://github.com/ArnaudCassan/microlensing.git
$ pip install --user microlensing/
```
(you may not need the `--user` option)

### Loading

```python
import microlensing
```

Modules included and examples
--------

### Loading a binary-lens caustic object

For a given `(s, q)` binary lens configuration, where `s` is the lens separation (in Einstein units) and `q` the lens mass ratio, create an object which contains the critical curves, the caustics, and if specified the  location of the cusps. It also contains the topology of the caustics (close, intermediate or wide). Plotting function are included to visualize the critical curves, caustics and cusps. There are a number of options detailed in `help(Caustics)`.

```python
from microlensing.caustics import Caustics
cc = Caustics(1.4, 0.1, N=400)
cc.pltcrit()
cc.pltcaus()

cc = Caustics(1.4, 0.1, cusp=True)
cc.pltcaus()
```

### Quadrupolar and Hexadecapolar approximations of binary-lens magnification 

Compute the binary-lens magnification for a: point-source (A0), quadrupole (A2) and Hexadecapole (A4) approximations using the method by <a href="http://adsabs.harvard.edu/abs/2017MNRAS.468.3993C">Cassan (2017)</a>.

```python
from microlensing import multipoles
multipoles.example()
```

### Q factors entering the expansion of z(zeta) of the lens equation

Compute the Q_(p-n,n) factors (p>=3) from <a href="http://adsabs.harvard.edu/abs/2017MNRAS.468.3993C">Cassan (2017)</a>.
This function is currently working for python 2 only.

```python
from microlensing import Rkp
Rkp.Q(3)
```

License
-------

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

<!-- Commentaire  --> 
