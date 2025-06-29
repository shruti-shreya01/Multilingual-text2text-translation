<<<<<<< HEAD
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='IndicTransToolkit',
    ext_modules=cythonize("IndicTransToolkit/processor.pyx"),
    zip_safe=False,
=======
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='IndicTransToolkit',
    ext_modules=cythonize("IndicTransToolkit/processor.pyx"),
    zip_safe=False,
>>>>>>> 61961dcee16732a8e6699d9f6350301d8ed6f5be
)