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
cdef cnp.dtype f32_dt = numpy.dtype(numpy.float32)
cdef cnp.float32_t inf = numpy.inf

# Some cdefs
cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
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
def get_denoised_patch(cnp.ndarray[float, ndim=3] to_denoise_array,
                       cnp.ndarray[size_t, ndim=1] array_index,
                       int vector_index,
                       cnp.ndarray[size_t, ndim=1] half_spatial_bandwidth,
                       cnp.ndarray[size_t, ndim=1] half_patch_size,
                       cnp.ndarray[size_t, ndim=1] full_patch_size,
                       float range_bandwidth,
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
        size_t x, y, z
        # > input image information
        size_t to_denoise_strides[3]
        size_t to_denoise_shape[3]
        float *to_denoise_ptr
        size_t *index_ptr
        # > patch parameters
        size_t lower_bound[3]
        size_t upper_bound[3]
        size_t nb_of_search_selements=0
        size_t patch_size=1
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
        patch_size *= full_patch_size[i]
        to_denoise_shape[i] = to_denoise_array.shape[i]
        
    for x from lower_bound[0] <= x <= upper_bound[0]:
        for y from lower_bound[1] <= y <= upper_bound[1]:
            for z from lower_bound[2] <= z <= upper_bound[2]:
                if (x != array_index[0] or y != array_index[1] or
                    z != array_index[2]):

                    search[nb_of_search_selements].index[0] = x
                    search[nb_of_search_selements].index[1] = y
                    search[nb_of_search_selements].index[2] = z
                    nb_of_search_selements += 1

    # Get flatten arrays
    to_denoise_ptr = <float *>to_denoise_array.data
    half_patch_size_ptr = <size_t *>half_patch_size.data
    index_ptr = <size_t *>array_index.data

    # Get patch around the current location.       
    central_patch = <float *>malloc(patch_size * sizeof(float))
    get_patch(index_ptr, to_denoise_ptr, to_denoise_shape, to_denoise_strides, 
              half_patch_size_ptr, patch_size, central_patch)
                
    # Go through all the search indices
    weights = <float *>malloc(nb_of_search_selements * sizeof(float))
    neighbor_patches = <float *>malloc(
        nb_of_search_selements *  patch_size * sizeof(float))
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
            get_patch(search[i].index, to_denoise_ptr, to_denoise_shape, 
                      to_denoise_strides, half_patch_size_ptr, patch_size,
                      neighbor_patches, i)

            # Compute the weight associated to the distance between the
            # central and neighbor patches
            weights[i] = exp(
                - patch_distance(central_patch, neighbor_patches, patch_size) /
                range_bandwidth)

            # Normalize the patch values
            for j from 0 <= j < patch_size:
                neighbor_patches[patch_size * i + j] /= weights[i]

        # Keep trace of the maximum weight and compute the weight sum
        # Get the result denoised patch
        for i from 0 <= i < nb_of_search_selements:

            # Maximum weight
            if weights[i] > wmax:
                wmax = weights[i]

            # Weight sum
            wsum += weights[i]

            # Get the result denoised patch
            for j from 0 <= j < patch_size:
                patch_ptr[j] += neighbor_patches[patch_size * i + j]

    # Free memory
    with nogil:
        for i from 0 <= i < nb_of_search_selements:
            free(search[i].index)
        free(search)
        free(lower_bound)
        free(upper_bound)
        free(central_patch)
        free(to_denoise_shape)
        free(neighbor_patches)
        free(weights)

    return patch, wsum, wmax


###############################################################################
# Intern functions
###############################################################################

cdef inline float ptr_value_from_array_index(float *array_ptr,
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
cdef void get_patch(size_t *array_index,
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
        size_t i
        # > array iterators
        size_t x, y, z
        # > patch lower and upper bounds
        size_t lower_bound[3]
        size_t upper_bound[3]
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
        for i  from 0 <= i < patch_size:
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
                        ptr_value_from_array_index(
                            array_ptr, neighbor_index, array_strides, 3))
                    i += 1

    # Free memory
    free(lower_bound)
    free(upper_bound)
    free(neighbor_index)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float patch_distance(float *patch1,
                                 float *patch2,
                                 size_t patch_size) nogil:
    """ Compute the distance between two patches.

    The metric used is the squared euclidean distance.

    Parameters
    ----------
    patch1: float[patch_size]
        a patch that has been flatten.
    patch2: float[patch_size]
        a patch that has been flatten.

    Returns
    -------
    dist: float
        the squared euclidean distance between the two patches.
    """
    cdef:
        # Intern parameters
        # > iterators
        size_t i
        # > squared euclidean distance
        float dist=0

    # Compute the squared euclidean distance.
    for i from 0 <= i < patch_size:
        dist += (patch1[i] - patch2[i]) * (patch1[i] - patch2[i])

    return dist
