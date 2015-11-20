from glob import glob
import os
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.command.build_clib import build_clib

library = ('primesieve', dict(
    sources=glob("lib/primesieve/src/primesieve/*.cpp"),
    include_dirs=["lib/primesieve/include"],
    language="c++",
    ))

# these become cython c compile time flags for conditional compilation
c_options = { 
    'use_numpy': 0
}

# try to import numpy
try:
    import numpy
    np_include_path = [numpy.get_include()]
    c_options['use_numpy'] = 1
except:
    numpy = None
    np_include_path = []

def generate_cython_c_config_file():
    print('Generate config.pxi')
    d = os.path.dirname(__file__)
    config_file_name = os.path.join(  d , 'primesieve', 'config.pxi')
    with open(config_file_name, 'w') as fd:
        for k, v in c_options.items():
            fd.write('DEF %s = %d\n' % (k.upper(), int(v)))

generate_cython_c_config_file()

if glob("primesieve/*.pyx"):
    from Cython.Build import cythonize
    EXT = 'pyx'                 # file extension to use if cython is installed
else:
    # fallback to compiled cpp
    cythonize = None
    EXT = 'cpp'                 # for distribution we include the generated cpp file, so use it for installation

extension = Extension(
        "primesieve",
        ["primesieve/primesieve.{}".format(EXT)],
        include_dirs=["lib/primesieve/include", "lib/primesieve/include/primesieve"] + np_include_path,
        language="c++",
        )

ext_modules = cythonize(extension) if cythonize else [extension]

def old_msvc(compiler):
    """Test whether compiler is msvc <= 9.0"""
    return compiler.compiler_type == "msvc" and hasattr(compiler, "_MSVCCompiler__version") and int(compiler._MSVCCompiler__version) <= 9

class build_clib_subclass(build_clib):
    """Workaround to add msvc_compat (stdint.h) for old msvc versions"""
    def build_libraries(self, libraries):
        if old_msvc(self.compiler):
            for lib_name, build_info in libraries:
                build_info['include_dirs'].append("lib/primesieve/src/msvc_compat")
                print(build_info)
        build_clib.build_libraries(self, libraries)

class build_ext_subclass(build_ext):
    """Workaround to add msvc_compat (stdint.h) for old msvc versions"""
    def build_extensions(self):
        if old_msvc(self.compiler):
            for e in self.extensions:
                e.include_dirs.append("lib/primesieve/src/msvc_compat")
        build_ext.build_extensions(self)

setup(
    name='primesieve',
    version="0.1.2",
    url = "https://github.com/hickford/primesieve-python",
    description="Fast prime number generator. Python bindings around C++ library primesieve",
    license = "MIT",
    libraries = [library],
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext_subclass, 'build_clib' : build_clib_subclass},
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
