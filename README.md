microlensing
======

<a href="https://travis-ci.org/ArnaudCassan/microlensing"><img src="https://travis-ci.org/ArnaudCassan/microlensing.svg?branch=master"></a>

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

### Loading

```python
import microlensing
```

Examples
--------

### Quadrupolar and Hexadecapolar approximations of binary-lens magnification 

Compute the binary-lens magnification for a: point-source (A0), quadrupole (A2) and Hexadecapole (A4) approximation using the method by <a href="http://adsabs.harvard.edu/abs/2017MNRAS.468.3993C">Cassan (2017)</a>:

```python
from microlensing import multipoles
multipoles.example()
```

### Q factors entering expansion of z(zeta) of the lens equation

Compute the Q_(p-n,n) factors (p>=3) <a href="http://adsabs.harvard.edu/abs/2017MNRAS.468.3993C">Cassan (2017)</a>:
```python
from microlensing import Rkp
Rkp.Q(3)
```

<!-- Commentaire  --> 
