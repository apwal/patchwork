#! /usr/bin/env python
################################################################################
# PATCHWORK - Copyright (C) 2014
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
################################################################################

""" This file contains/defines parameters for the PATCHWORK software.
"""

_version_major = 0
_version_minor = 0
_version_micro = 1

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(_version_major, _version_minor, _version_micro)

classifiers = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

description = "PatchWork"

long_description = """
=========
PatchWork
==========

PatchWork description.
"""

# versions for dependencies
SPHINX_MIN_VERSION = 1.0
NUMPY_MIN_VERSION = '1.3'
SCIPY_MIN_VERSION = '0.7.2'

# Main setup parameters
NAME = "patchwork"
MAINTAINER = "apwal"
MAINTAINER_EMAIL = "apwal"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "apwal"
DOWNLOAD_URL = ""
LICENSE = "CeCILL-B"
CLASSIFIERS = classifiers
AUTHOR = "apwal"
AUTHOR_EMAIL = "apwal"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PROVIDES = ["patchwork"]
REQUIRES = ["numpy>={0}".format(NUMPY_MIN_VERSION),
            "scipy>={0}".format(SCIPY_MIN_VERSION)]
EXTRA_REQUIRES = {"doc": ["sphinx>=1.0"]}
