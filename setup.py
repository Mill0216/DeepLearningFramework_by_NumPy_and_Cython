from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

name = 'Layers'

ext = Extension(name, sources=["{}.pyx".format(name)], include_dirs=['.', numpy.get_include()])
setup(name=name, ext_modules=cythonize([ext]))