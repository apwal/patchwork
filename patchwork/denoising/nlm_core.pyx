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

# PATCHWORK import
include "../tools/patch_core.pyx"

# Float 32 dtype for casting
cdef cnp.dtype f32_dt = numpy.dtype(numpy.float32)
cdef cnp.float32_t inf = numpy.inf

# Some cdefs
cdef extern from "stdlib.h" nogil:
    ctypedef long int size_t
cdef extern from "math.h" nogil:
    float sqrt(float x)
    float exp(float x)

# Define a track structure
ctypedef struct Element:
    size_t index[3]


###############################################################################
# Python callable functions
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_average_patch(cnp.ndarray[float, ndim=3] to_denoise_array,
                      cnp.ndarray[long, ndim=1] array_index,
                      cnp.ndarray[long, ndim=1] half_spatial_bandwidth,
                      cnp.ndarray[long, ndim=1] half_patch_size,
                      cnp.ndarray[long, ndim=1] full_patch_size,
                      float range_bandwidth,
                      int nb_of_threads=1):
    """ Method that compute the denoised patch at a specific location.

    Parameters
    ----------
    to_denoise_array: 3-dim array
        the image to denoise.
    array_index: 1-dim array
        the patch central location in the array to denoise.
    half_spatial_bandwidth: 1-dim array with 3 items
        the half size of the search region.
    half_patch_size: 1-dim array with 3 items
        the half size of the patch array.
    full_patch_size: 1-dim array with 3 items
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
        size_t i, j
        # > array iterators
        long x, y, z
        # > input image information
        size_t to_denoise_strides[3]
        size_t to_denoise_shape[3]
        float *to_denoise_ptr
        size_t *index_ptr
        # > patch parameters
        int lower_bound[3]
        int upper_bound[3]
        size_t nb_of_search_selements=0
        size_t patch_size=1
        size_t neighbor_max_size=1
        float *patch_ptr
        size_t *half_patch_size_ptr
        float *central_patch
        float *neighbor_patches
        float *weights
        # > maximum weight of patches
        float wmax = 0
        # > sum of weights: used for normalization
        float wsum = 0
        # > store the search indices
        Element *search

    # Allocate output patch
    patch = numpy.zeros(full_patch_size, dtype=numpy.single)
   
    # Create an appropriate search region around the current voxel index
    # Get input image information
    for i from 0 <= i < 3:
        lower_bound[i] = array_index[i] - half_spatial_bandwidth[i]
        upper_bound[i] = array_index[i] + half_spatial_bandwidth[i]
        to_denoise_strides[i] = to_denoise_array.strides[i]
        patch_size *= full_patch_size[ i]
        neighbor_max_size *= (2 * half_spatial_bandwidth[i] + 1)
        to_denoise_shape[i] = to_denoise_array.shape[i]
    
    search = <Element *>malloc(neighbor_max_size * sizeof(Element))
    for x from lower_bound[0] <= x <= upper_bound[0]:
        for y from lower_bound[1] <= y <= upper_bound[1]:
            for z from lower_bound[2] <= z <= upper_bound[2]:

                # Do not consider the current point
                if (x != array_index[0] or y != array_index[1] or
                    z != array_index[2]):

                    # Check for image boudaries
                    if (x >= 0 and y >= 0 and z >= 0 and 
                        x < to_denoise_shape[0] and y < to_denoise_shape[1] and 
                        z < to_denoise_shape[2]):

                        # Add the new point in the search region
                        search[nb_of_search_selements].index[0] = x
                        search[nb_of_search_selements].index[1] = y
                        search[nb_of_search_selements].index[2] = z

                        # Keep trace of the search region size
                        nb_of_search_selements += 1

    # Get flatten arrays
    to_denoise_ptr = <float *>to_denoise_array.data
    half_patch_size_ptr = <size_t *>half_patch_size.data
    index_ptr = <size_t *>array_index.data

    # Get patch around the current location.       
    central_patch = <float *>malloc(patch_size * sizeof(float))
    _get_patch(index_ptr, to_denoise_ptr, to_denoise_shape, to_denoise_strides, 
               half_patch_size_ptr, patch_size, central_patch)
                
    # Go through all the search indices
    weights = <float *>malloc(nb_of_search_selements * sizeof(float))
    neighbor_patches = <float *>malloc(
        nb_of_search_selements * patch_size * sizeof(float))
    patch_ptr = <float *>malloc(patch_size * sizeof(float))
    for i from 0 <= i < patch_size:
        patch_ptr[i] = 0
    with nogil, cython.boundscheck(False), cython.wraparound(False):

        # Search parallel loop
        for i in parallel.prange(0, nb_of_search_selements, schedule="static",
                                 num_threads=nb_of_threads):

            # Optimize the speed of the patch search
            #if self.use_optimized_strategy: 
            #    is_valid_neighbor = self._check_speed(
            #        tuple(index), tuple(neighbor_index))
            #else:
            #    is_valid_neighbor = True

            # If we are dealing with a valid neighbor
            #if is_valid_neighbor:

            # Create the neighbor patch
            _get_patch(search[i].index, to_denoise_ptr, to_denoise_shape, 
                       to_denoise_strides, half_patch_size_ptr, patch_size,
                     neighbor_patches, i)

            # Compute the weight associated to the distance between the
            # central and neighbor patches
            weights[i] = exp(- _patch_distance(
                central_patch, neighbor_patches, patch_size, i) / range_bandwidth)

        # Keep trace of the maximum weight and compute the weight sum
        # Get the result denoised patch
        for i from 0 <= i < nb_of_search_selements:

            # Maximum weight
            if weights[i] > wmax:
                wmax = weights[i]

            # Weight sum
            wsum += weights[i]

            # Get the result denoised normalized patch
            for j from 0 <= j < patch_size:
                patch_ptr[j] += neighbor_patches[patch_size * i + j] * weights[i]

    # Copy result patch
    patch = patch.flatten()
    for i from 0 <= i < patch_size:
        patch[i] = patch_ptr[i]
    patch = patch.reshape(full_patch_size)

    # Free memory
    with nogil:
        free(search)
        free(central_patch)
        free(neighbor_patches)
        free(weights)

    return patch, wsum, wmax
