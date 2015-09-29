from setuptools import setup, Extension
from glob import glob
from distutils.command.build_ext import build_ext
from distutils.command.build_clib import build_clib

sources = glob("lib/primesieve/src/primesieve/*.cpp")

if glob("primesieve/*.pyx"):
    from Cython.Build import cythonize
    sources.append("primesieve/primesieve.pyx")
else:
    # fallback to compiled cpp
    sources.append("primesieve/primesieve.cpp")
    cythonize = None

extension = Extension(
        "primesieve",
        sources,
        include_dirs=["lib/primesieve/include", "lib/primesieve/include/primesieve"],
        language="c++",
        )

ext_modules = cythonize(extension) if cythonize else [extension]

def old_msvc(compiler):
    """Test whether compiler is msvc <= 9.0"""
    return compiler.compiler_type == "msvc" and hasattr(compiler, "_MSVCCompiler__version") and int(compiler._MSVCCompiler__version) <= 9

extra_compile_args_by_compiler = {
    "unix": ["-fopenmp"],
    "cygwin": ["-fopenmp"],
    "mingw32": ["-fopenmp"],
    "msvc": ["/openmp"],
}

class build_ext_subclass(build_ext):
    """Workaround to add msvc_compat (stdint.h) for old msvc versions"""
    def build_extensions(self):
        for e in self.extensions:
            e.extra_compile_args.extend(extra_compile_args_by_compiler.get(self.compiler.compiler_type, []))
            if old_msvc(self.compiler):
                e.include_dirs.append("lib/primesieve/src/msvc_compat")
        build_ext.build_extensions(self)

setup(
    name='primesieve',
    version="0.1.2",
    url = "https://github.com/hickford/primesieve-python",
    description="Fast prime number generator. Python bindings around C++ library primesieve",
    license = "MIT",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext_subclass},
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
