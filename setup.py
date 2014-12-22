#! /usr/bin/env python
################################################################################
# PATCHWORK - Copyright (C) 2014
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
################################################################################

""" Command to build extension:
> python setup.py build_ext --inplace
"""

# System include
from ez_setup import use_setuptools
use_setuptools()
import os
from setuptools import setup, find_packages, Extension
import numpy
from Cython.Distutils import build_ext


cmdclass = {"build_ext": build_ext}

ext_modules = []
for modulename, other_sources, language in (
        ("patchwork.denoising.nlm_core", [], "c"), ):
    pyx_src = os.path.join(*modulename.split(".")) + ".pyx"
    ext_modules.append(Extension(modulename, [pyx_src] + other_sources,
                                 language=language,
                                 include_dirs=[numpy.get_include(), "src"],
                                 extra_compile_args=['-fopenmp'],
                                 extra_link_args=['-fopenmp']))


release_info = {}
execfile(os.path.join("patchwork", "info.py"), release_info)


setup(
    name=release_info["NAME"],
    description=release_info["DESCRIPTION"],
    long_description=release_info["LONG_DESCRIPTION"],
    license=release_info["LICENSE"],
    classifiers=release_info["CLASSIFIERS"],
    author=release_info["AUTHOR"],
    author_email=release_info["AUTHOR_EMAIL"],
    version=release_info["VERSION"],
    url=release_info["URL"],
    packages=find_packages(exclude="doc"),
    platforms=release_info["PLATFORMS"],
    extras_require=release_info["EXTRA_REQUIRES"],
    install_requires=release_info["REQUIRES"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
