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

# PATCHWORK import
from .coordinate import vector_to_array_index

# Get the logger
logger = logging.getLogger(__file__)


def normalize_patch_size(half_patch_size, spacing):
    """ Normalize patch size according to the image anisotropy.

    Parameters
    ----------
    half_patch_size: int
        the half size of the patch in voxel.
    spacing: array
        the image spacing used for normalization.

    Returns
    -------
    notmalize_half_patch_size: array (x, y, z)
        the half size of the denoising patches in voxel.
    normalize_full_patch_size: array (x, y, z)
        the full size of the denoising patches in voxel: 2 * half_patch_size + 1.
    """
    # Information message
    logger.info("Normalize patch size according to the image anisotropy.")

    # Compute the normlize patch size
    min_spacing = numpy.min(spacing)
    normalize_half_patch_size = numpy.cast[numpy.int](
        0.5 + half_patch_size * min_spacing / spacing)
    normalize_full_patch_size = 2 * normalize_half_patch_size + 1

    # Information message
    logger.info("The half patch size is '%s'", normalize_half_patch_size)

    return normalize_half_patch_size, normalize_full_patch_size


def patch_distance(patch1, patch2):
    """ Compute the distance between two patches.

    The metric used is the squared euclidean distance.

    Parameters
    ----------
    patch1: array (N,)
        a patch that has been flatten.
    patch2: array (N,)
        a patch that has been flatten.

    Returns
    -------
    dist: float
        the distance between the two patches.
    """
    # Check that the two patches have the same size
    if patch1.size != patch2.size:
        raise ValueError("Unexpected patch size in distance computation.")

    # Compute the squared euclidean distance.
    dist = numpy.sum((patch1 - patch2)**2)

    return dist


def get_patch(index, array, patch_shape, dtype=numpy.single):
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
    # Intern parameters
    shape = array.shape
    half_patch_shape = numpy.cast[numpy.int](patch_shape / 2)

    # Create an appropriate patch region around the current voxel index
    # neglecting pixels close to the boundaries
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
      
  
def get_patch_elements(index, shape, half_patch_shape): 
    """ Method that compute the indices of a patch at a specific location.

    Parameters
    ----------
    index: array
        the patch central location.
    shape: array
        the shape of the array from which the patch is extracted.
    half_patch_size: int
        the half patch size.

    Returns
    -------
    patch_elements: list of 2-uplet
        the first element is index of the patch, and the second element
        is the corresponding index in the image index space.
    """
    # Create an appropriate patch region around the current voxel index
    # neglecting pixels close to the boundaries
    lower_bound = index - half_patch_shape
    upper_bound = index + half_patch_shape

    # Check that the patch is inside the image
    if (lower_bound < 0).any() or (upper_bound >= shape).any():
        logger.debug("Reach the border when creating patch at position "
                     "'%s'.", index)
        return []

    # Get the patch
    patch_elements = []
    for cntx, x in enumerate(range(lower_bound[0], upper_bound[0] + 1)):
        for cnty, y in enumerate(range(lower_bound[1], upper_bound[1] + 1)):
            for cntz, z in enumerate(range(lower_bound[2], upper_bound[2] + 1)):
                patch_elements.append(((cntx, cnty, cntz), (x, y, z)))

    return patch_elements


def patch_mean_variance(array, mask_array, patch_shape):
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
    # Information message
    logger.info("Optimized mode, compute mean and variance image.") 

    # Allocate the two arrays
    mean = numpy.zeros(array.shape, dtype=numpy.single)
    variance  = numpy.zeros(array.shape, dtype=numpy.single)

    # Go through all the voxels
    for vector_index in range(array.size):

        # Convert vector index to array index
        index = vector_to_array_index(vector_index, array)

        # Check if have to compute this voxel
        if mask_array[tuple(index)] > 0:

            # Get the surrounding patch
            patch = get_patch(index, array, patch_shape)

            # Compute the local mean and variance
            local_mean = numpy.mean(patch)
            local_variance = numpy.mean(patch**2) - local_mean**2

            # Store the computed values
            mean[tuple(index)] = local_mean
            variance[tuple(index)] = local_variance

    return mean, variance


