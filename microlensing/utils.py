"""microlensing related utilities"""

# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

import numpy as np
import timeit
import os
import logging
from logging import info as printi
from logging import debug as printd
from logging import warning as printw

def verbosity(level, filename=None):
    """Set verbosity flag
    """
    # set I/O shell display
    tend, tun, tit = "\033[0m", "\033[0m\033[31;1m", "\033[0m\033[3m"
    
    if level == 'NONE':
        verbose = logging.CRITICAL
        logging.basicConfig(format='%(message)s', level=verbose, filename=filename)
    elif level == 'WARNING':
        verbose = logging.WARNING
        logging.basicConfig(format=tun + 'Warning/Error : ' + tend + '%(message)s', level=verbose, filename=filename)
    elif level == 'INFO':
        verbose = logging.INFO
        logging.basicConfig(format='%(message)s', level=verbose, filename=filename)
    elif level == 'DEBUG':
        verbose = logging.DEBUG
        logging.basicConfig(format=tun + '[%(process)d] ' + tend + '%(message)s', level=verbose, filename=filename)
    else:
        raise("error level does not exist: correct options are 'NONE', 'WARNING', 'INFO' and 'DEBUG'")

def checkandtimeit():
    """Check and time function
        
        Usage
        -----
        @checkandtimeit()
        def function(*args, **kwargs)
            (...)
        """
    def _deco_verif(fonction):
        # set I/O shell display
        tend, tcol, tit = "\033[0m", "\033[47m", "\033[3m"
        def _new_verif(*args, **kwargs):
            printi(tcol + "Begin execution:" + tend + tit + " {}".format(fonction.__name__) + tend)
            t1 = timeit.default_timer()
            ret = fonction(*args, **kwargs)
            t2 = timeit.default_timer()
            printi(tcol + "End execution:" + tend + tit + " {}".format(fonction.__name__) + tend + " (executed in {:.2e}s)".format(t2-t1))
            return ret
        return _new_verif
    return _deco_verif

def obsolete(func):
    """Mark function as obsolete
        
        Usage
        -----
        @obsolete
        def function(*args, **kwargs)
            (...)
        """
    def mfunc():
        raise RuntimeError("Function '{}' is obsolete".format(func.__name__))
    return mfunc

