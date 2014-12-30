#! /usr/bin/env python
################################################################################
# PATCHWORK - Copyright (C) 2014
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
################################################################################

# System import
import logging
import numpy

# Get the logger
logger = logging.getLogger(__file__)


def vector_to_array_index(vector_index, array):
    """ Transform a vector index to the corresponding array index.

    Parameters
    ----------
    vector_index: int
        the vector index.
    array: array
        the input array data.

    Returns
    -------
    array_index: array
        the corresponding vector index.
    """
    return numpy.asarray(numpy.unravel_index(vector_index, array.shape))


def array_to_vector_index(array_index, array):
    """ Transform an array index to the corresponding vector index.

    Parameters
    ----------
    array_index: array
        the corresponding vector index.
    array: array
        the input array data.

    Returns
    -------
    vector_index: int
        the vector index.
    """
    offset = numpy.sum(numpy.asarray(array_index) * array.strides)
    return offset / array.itemsize
