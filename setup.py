from setuptools import setup, Extension
from glob import glob
from setuptools import distutils
from distutils.command.build_ext import build_ext

class build_ext_subclass(build_ext):
    def build_extensions(self):
        print(self)
        print(self.compiler)
        print(self.compiler.compiler_type)
        build_ext.build_extensions(self)

library = ('primesieve', dict(
    sources=glob("lib/primesieve/src/primesieve/*.cpp"),
    include_dirs=["lib/primesieve/include"],
    language="c++",
    ))

if glob("primesieve/*.pyx"):
    from Cython.Build import cythonize
else:
    # fallback to compiled cpp
    cythonize = None

extension = Extension(
        "primesieve",
        ["primesieve/primesieve.pyx"] if cythonize else ["primesieve/primesieve.cpp"],
        include_dirs=["lib/primesieve/include", "lib/primesieve/include/primesieve"],
        language="c++",
        )

ext_modules = cythonize(extension) if cythonize else [extension]

setup(
    name='primesieve',
    version="0.1.0",
    url = "https://github.com/hickford/primesieve-python",
    description="Fast prime number generator. Python bindings around C++ library primesieve",
    license = "MIT",
    libraries = [library],
    ext_modules = ext_modules,
    classifiers=[
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    ],
)
