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
cimport openmp
from cython cimport parallel
from libc.stdlib cimport abort, malloc, free, realloc

# Float 32 dtype for casting
cdef cnp.dtype f32_dt = np.dtype(np.float32)
cdef cnp.float32_t inf = np.inf

# Some cdefs
cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
cdef extern from "math.h" nogil:
    float sqrt(float x)

# Define a track structure
ctypedef struct Element:
    size_t index[3]


###############################################################################
# Python callable functions
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_denoised_patch(cnp.ndarray[float, ndim=3] to_denoise_array,
                       cnp.ndarray[float, ndim=1] array_index,
                       int vector_index,
                       half_spatial_bandwidth,
                       full_patch_size,
                       int nb_of_threads=1):
    """ Method that compute the denoised patch at a specific location.

    Parameters
    ----------
    to_denoise_array: 3-dim array
        the image to denoise.
    array_index: 1-dim array
        the patch central location in the array to denoise.
    vector_index: int
        the patch central location in the flatten array to denoise.
    full_patch_size: 3-uplet
        the half size of the patch array.
    full_patch_size: 3-uplet
        the size of the patch array.
    range_bandwidth: float
        the global bandwidth range.
    nb_of_threads: int (default 1)
        the number of threads to use.

    Returns
    -------
    patch: aray
        the denoised patch.
    weight: float
        the patch power (ie., the sum of all the weights associated to
        the neighbor patches).
    """
    cdef:
        # Intern parameters
        # > iterators
        size_t i
        # > array iterators
        size_t x, y, z
        # > input image information
        size_t to_denoise_strides[3]
        float *to_denoise_ptr
        # > patch lower and upper bounds
        size_t lower_bound[3]
        size_t upper_bound[3]
        size_t nb_of_search_selements=0
        # > maximum weight of patches
        float wmax = 0
        # > sum of weights: used for normalization
        float wsum = 0
        # > store the search indices
        Element *search

    # Allocate the patch result
    patch = numpy.zeros(full_patch_size, dtype=numpy.single)
    
    # Create an appropriate search region around the current voxel index
    for i from 0 <= i < 3:
        lower_bound[i] = array_index[i] - half_spatial_bandwidth[i]
        upper_bound[i] = array_index[i] + half_spatial_bandwidth[i]
        to_denoise_strides[i] = to_denoise_array.strides[i]
        
    for x from lower_bound[0] <= x <= upper_bound[0]:
        for y from lower_bound[1] <= y <= upper_bound[1]:
            for z from lower_bound[2] <= z <= upper_bound[2]:
                search[nb_of_search_selements].index[0] = x
                search[nb_of_search_selements].index[1] = y
                search[nb_of_search_selements].index[2] = z
                nb_of_search_selements += 1

    # Get patch around the current location.
    to_denoise_ptr = <float *>to_denoise_array.data
    central_patch = get_patch(
        index, to_denoise_ptr, to_denoise_strides, full_patch_size)
                
    # Go through all the search indices
    with nogil, cython.boundscheck(False), cython.wraparound(False):

        # Search parallel loop
        for i in parallel.prange(0, nb_of_search_selements, schedule="static",
                                 num_threads=nb_of_threads):

            # Create the neighbor patch
            neighbor_patch = get_patch(
                neighbor_index, to_denoise_array, full_patch_size)

            # Compute the weight associated to the distance between the
            # central and neighbor patches
            weight = numpy.exp(
                - patch_distance(central_patch, neighbor_patch) / range_bandwidth)

            # Keep trace of the maximum weight
            if weight > wmax:
                wmax = weight

            # Update the weight sum
            wsum += weight

            # Update the result denoised patch
            patch += neighbor_patch / weight






    for neighbor_index in search_elements:

        # Optimize the speed of the patch search
        if self.use_optimized_strategy: 
            is_valid_neighbor = self._check_speed(
                tuple(index), tuple(neighbor_index))
        else:
            is_valid_neighbor = True

        # If we are dealing with a valid neighbor
        if is_valid_neighbor:

            # Create the neighbor patch
            neighbor_patch = get_patch(
                neighbor_index, self.to_denoise_array, self.full_patch_size)

            # Compute the weight associated to the distance between the
            # central and neighbor patches
            weight = numpy.exp(
                - patch_distance(central_patch, neighbor_patch) /
                self.range_bandwidth)

            # Keep trace of the maximum weight
            if weight > wmax:
                wmax = weight

            # Update the weight sum
            wsum += weight

            # Update the result denoised patch
            patch += neighbor_patch / weight

    # Free memory
    with nogil:
        for i from 0 <= i < nb_of_search_selements:
            free(search[i].index)
        free(search)
        free(lower_bound)
        free(upper_bound)

    return patch, wsum, wmax


###############################################################################
# Intern functions
###############################################################################

cdef inline float ptr_value_from_array_index(float *array_ptr,
                                             unsigned int *array_index,
                                             unsigned int *array_strides,
                                             int nb_dims) nogil:
    """ Get the value of a buffer array at a specific array index.
    """
    cdef:
        unsigned int i, offset=0
    for i from 0 <= i < nb_dims:
        offset += array_index[i] * array_strides[i]
    offset = <unsigned int>(<float>offset / sizeof(float))
    return array_ptr[offset]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef get_patch(size_t *array_index, float *array_ptr, size_t *half_patch_shape,
               size_t *array_shape):
    """ Get a patch at a specific index.

    If the surrounding patch goes beyound the image boudaries, return an empty
    patch.

    Parameters
    ----------
    array_index: unsigned long[3]
        a valid array index.
    array_ptr: float[*]
        the array from which the patch is extracted.
    half_patch_shape: unsigned long[3]
        the half patch shape.
    array_shape: unsigned long[3]
        the shape of the array.
    
    Returns
    -------
    patch: float[patch_shape]
        the patch at the specified position.
    """
    # Create an appropriate patch region around the current voxel index
    # neglecting pixels close to the boundaries
    for i from 0 <= i < 3:
        lower_bound[i] = array_index[i] - half_spatial_bandwidth[i]
        upper_bound[i] = array_index[i] + half_spatial_bandwidth[i]
        to_denoise_strides[i] = to_denoise_array.strides[i]
        
    for x from lower_bound[0] <= x <= upper_bound[0]:
        for y from lower_bound[1] <= y <= upper_bound[1]:
            for z from lower_bound[2] <= z <= upper_bound[2]:

    lower_bound = index - half_patch_shape
    upper_bound = index + half_patch_shape

    # Check that the patch is inside the image
    if (lower_bound < 0).any() or (upper_bound >= shape).any():
        logger.debug("Reach the border when creating patch at position "
                     "'%s'.", index)
        return numpy.zeros(patch_shape, dtype=dtype)

    # Get the patch
    return array[lower_bound[0]: upper_bound[0] + 1,
                 lower_bound[1]: upper_bound[1] + 1,
                 lower_bound[2]: upper_bound[2] + 1]
