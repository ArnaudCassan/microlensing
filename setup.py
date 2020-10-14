name = 'microlensing'

import os, sys, platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

if platform.system() == 'Windows':
    compile_extra_args = ['/std:c++latest', '/EHsc']
elif platform.system() == 'Darwin':
    compile_extra_args = ['-std=c++11', '-mmacosx-version-min=10.9']
    link_extra_args = ['-stdlib=libc++', '-mmacosx-version-min=10.9']
else:
    compile_extra_args = []
    link_extra_args = []
    
extensions = [Extension('microlensing.mismap.vbb.vbb', sources=['microlensing/mismap/vbb/vbb.pyx'], language='c++', extra_compile_args=compile_extra_args, extra_link_args=link_extra_args)]

setup(name = name, ext_modules = cythonize(extensions))
   
pjoin = os.path.join
here = os.path.abspath(os.path.dirname(__file__))

packages = []
for d, _, _ in os.walk(pjoin(here, name)):
    if os.path.exists(pjoin(d, '__init__.py')):
        packages.append(d[len(here)+1:].replace(os.path.sep, '.'))

version_ns = {}
with open(pjoin(here, name, '_version.py')) as f:
    exec(f.read(), {}, version_ns)

setup_args = dict(
    name            = name,
    version         = version_ns['__version__'],
    packages        = packages,
    description     = 'Gravitational microlensing',
    long_description= 'Gravitational microlensing',
    author          = 'Arnaud Cassan',
    author_email    = 'arnaud.cassan@iap.fr',
    url             = 'https://github.com/ArnaudCassan/microlensing.git',
    license         = 'MIT License',
    platforms       = 'Linux, Mac OS X, Windows',
    keywords        = ['Astronomy', 'Astrophysics', 'Microlensing', 'Science', 'Exoplanets'],
    classifiers     = [
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)

if 'develop' in sys.argv or any(a.startswith('bdist') for a in sys.argv):
    import setuptools

setuptools_args = {}
install_requires = setuptools_args['install_requires'] = []# ['numpy', 'h5py', 'scipy', 'pandas', 'bokeh', 'matplotlib', 'sympy', 'mpmath', 'astropy', 'multiprocessing']

extras_require = setuptools_args['extras_require'] = {}

if 'setuptools' in sys.modules:
    setup_args.update(setuptools_args)

if __name__ == '__main__':
    setup(**setup_args)
