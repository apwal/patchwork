#! /usr/bin/env python
################################################################################
# PATCHWORK - Copyright (C) 2014
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
################################################################################

# System import
cimport cython
import numpy as numpy
cimport numpy as cnp
from libc.stdlib cimport abort, malloc, free, realloc

# Some cdefs
cdef extern from "stdlib.h" nogil:
    ctypedef long int size_t


###############################################################################
# Python callable functions
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_patch(cnp.ndarray[long, ndim=1] index,
              cnp.ndarray[float, ndim=3] array,
              patch_shape,
              dtype=numpy.single):
    """ Get a patch at a specific index.

    If the surrounding patch goes beyound the image boudaries, return an empty
    patch.

    Parameters
    ----------
    index: array
        a valid array index.
    array: array
        the array from which the patch is extracted.
    patch_shape: array
        the patch shape.
    dtype: type
        the output patch type.
    
    Returns
    -------
    patch: array (patch_shape, )
        the patch at the specified position.
    """
    cdef:
        # Intern parameters
        # > input image information
        size_t array_strides[3]
        size_t array_shape[3]
        float *array_ptr
        size_t *index_ptr
        # > patch parameters
        size_t half_patch_shape[3]
        size_t patch_size=1
        float *cpatch

    # Allocate output patch
    patch = numpy.zeros(patch_shape, dtype=dtype)

    # Get input image information
    for i from 0 <= i < 3:
        array_strides[i] = array.strides[i]
        patch_size *= patch_shape[i]
        array_shape[i] = array.shape[i]
        half_patch_shape[i] = int((patch_shape[i] - 1) / 2)

    # Get flatten arrays
    array_ptr = <float *>array.data
    index_ptr = <size_t *>index.data

    # Get patch around the current location.       
    cpatch = <float *>malloc(patch_size * sizeof(float))
    _get_patch(index_ptr, array_ptr, array_shape, array_strides, 
               half_patch_shape, patch_size, cpatch)

    # Copy result patch
    patch = patch.flatten()
    for i from 0 <= i < patch_size:
        patch[i] = cpatch[i]
    patch = patch.reshape(patch_shape)

    # Free memory
    with nogil:
        free(cpatch)

    return patch


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def patch_mean_variance(cnp.ndarray[float, ndim=3] array,
                        cnp.ndarray[float, ndim=3] mask_array,
                        patch_shape):
    """ Compute the mean and the variance of the image to denoise.

    Parameters
    ----------
    array: array
        the input array.
    mask_array: array
        a binary mask to apply during the mean and variance computation.
    patch_shape: array
        the patch shape.

    Returns
    -------
    mean: array
        the image mean.
    variance: array
        the image variance.
    """
    cdef:
        # Intern parameters
        # > iterators
        long x, y, z
        long i
        # > input image information
        size_t array_strides[3]
        size_t array_shape[3]
        float *array_ptr
        size_t index[3]
        # > patch parameters
        size_t half_patch_shape[3]
        size_t patch_size=1
        float *cpatch
        float local_mean, local_variance

    # Allocate the two output arrays
    shape = (array.shape[0], array.shape[1], array.shape[2])
    mean = numpy.zeros(shape, dtype=numpy.single)
    variance  = numpy.zeros(shape, dtype=numpy.single)

    # Get input image information
    for i from 0 <= i < 3:
        array_strides[i] = array.strides[i]
        patch_size *= patch_shape[i]
        array_shape[i] = array.shape[i]
        half_patch_shape[i] = int((patch_shape[i] - 1) / 2)

    # Allocate the patch
    cpatch = <float *>malloc(patch_size * sizeof(float))

    # Get flatten arrays
    array_ptr = <float *>array.data

    # Go through all the voxels
    for x from 0 <= x < array_shape[0]:
        for y from 0 <= y < array_shape[1]:
            for z from 0 <= z < array_shape[2]:

                # Check if have to compute this voxel
                if mask_array[x, y, z] > 0:

                    # Get the surrounding patch
                    index[0] = x; index[1] = y; index[2] = z
                    _get_patch(index, array_ptr, array_shape, array_strides, 
                               half_patch_shape, patch_size, cpatch)

                    # Compute the local mean and variance
                    local_mean = 0
                    local_variance = 0
                    for i from 0 <= i < patch_size:
                        local_mean += cpatch[i]
                        local_variance += cpatch[i] * cpatch[i]
                    local_mean /= float(patch_size)
                    local_variance /= float(patch_size)
                    local_varaince -= local_mean * local_mean

                    # Store the computed values
                    mean[x, y, z] = local_mean
                    variance[x, y, z] = local_variance

    return mean, variance


###############################################################################
# Intern functions
###############################################################################

cdef inline float _ptr_value_from_array_index(float *array_ptr,
                                              size_t *array_index,
                                              size_t *array_strides,
                                              int nb_dims) nogil:
    """ Get the value of a buffer array at a specific array index.

    Parameters
    ----------
    array_ptr: float[*]
        the array from which we want to get a value element.
    array_index: unsigned long[3]
        a valid array index.
    array_strides: unsigned long[3]
        the size of each array dimension.
    nb_dims: int
        the array number of dimensions.

    Returns
    -------
    array_value: float
        the desired array value.
    """
    cdef:
        unsigned int i, offset=0

    # Compute the memory offset of the desired element
    for i from 0 <= i < nb_dims:
        offset += array_index[i] * array_strides[i]
    offset = <unsigned int>(<float>offset / sizeof(float))

    return array_ptr[offset]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _get_patch(size_t *array_index,
                     float *array_ptr,
                     size_t *array_shape,
                     size_t *array_strides,
                     size_t *half_patch_shape,
                     size_t patch_size,
                     float *patch,
                     size_t patch_offset=0) nogil:
    """ Get a patch at a specific index.

    If the surrounding patch goes beyound the image boudaries, return an empty
    patch.

    Parameters
    ----------
    array_index: unsigned long[3]
        a valid array index.
    array_ptr: float[*]
        the array from which the patch is extracted.
    array_shape: unsigned long[3]
        the shape of the array.
    array_strides: unsigned long[3]
        the size of each array dimension.
    half_patch_shape: unsigned long[3]
        the half patch shape.
    patch_size: unsigned long
        the patch number of elements.
    patch: float[patch_size]
        the result patch at the specified position.
    patch_offset: unsinged long
         the patch pointer offset.
    """
    cdef:
        # Intern parameters
        # > iterators
        size_t i, j
        # > array iterators
        size_t x, y, z
        # > patch lower and upper bounds
        int lower_bound[3]
        int upper_bound[3]
        # > check if the patch neighborhood is valid
        int is_valid=1
        # > neighbor patch index
        size_t neighbor_index[3]

    # Create an appropriate patch region around the current voxel index
    # neglecting pixels close to the boundaries
    for i from 0 <= i < 3:
        lower_bound[i] = array_index[i] - half_patch_shape[i]
        upper_bound[i] = array_index[i] + half_patch_shape[i]

        # Check that the patch is inside the image
        if lower_bound[i] < 0 or upper_bound[i] >= array_shape[i]:
            is_valid = 0
            break

    # If the patch is not inside the image return an empty patch
    if is_valid == 0:
        for i from 0 <= i < patch_size:
            patch[patch_size * patch_offset + i] = 0
    else:
        i = 0
        for x from lower_bound[0] <= x <= upper_bound[0]:
            for y from lower_bound[1] <= y <= upper_bound[1]:
                for z from lower_bound[2] <= z <= upper_bound[2]:
                    neighbor_index[0] = x
                    neighbor_index[1] = y
                    neighbor_index[2] = z
                    patch[patch_size * patch_offset + i] = (
                        _ptr_value_from_array_index(
                            array_ptr, neighbor_index, array_strides, 3))
                    i += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float _patch_distance(float *patch1,
                                  float *patch2,
                                  size_t patch_size,
                                  size_t patch2_offset=0) nogil:
    """ Compute the distance between two patches.

    The metric used is the squared euclidean distance.

    Parameters
    ----------
    patch1: float[patch_size]
        a patch that has been flatten.
    patch2: float[patch_size]
        a patch that has been flatten.
    patch_size: unsigned long
        the patch number of elements.
    patch2_offset: unsinged long
         the patch2 pointer offset.

    Returns
    -------
    dist: float
        the squared euclidean distance between the two patches.
    """
    cdef:
        # Intern parameters
        # > iterators
        size_t i
        size_t offset
        # > squared euclidean distance
        float dist=0

    # Compute the squared euclidean distance.
    for i from 0 <= i < patch_size:
        offset = patch_size * patch2_offset + i
        dist += (patch1[i] - patch2[offset]) * (patch1[i] - patch2[offset])

    return dist
