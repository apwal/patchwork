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
import scipy.signal

# Patchwork import
from patchwork.tools import (
    get_patch, patch_distance, normalize_patch_size, get_patch_elements)
from patchwork.tools import vector_to_array_index, array_to_vector_index
from patchwork.denoising.nlm_core import get_average_patch

# Get the logger
logger = logging.getLogger(__file__)


class NLMDenoising(object):
    """ Patch denoising.

    Denoise an image using non-local mean (NLM) algorithm.

    Attributes
    ----------
    to_denoise_array: array
        an input n dimensional array to process containing the image
        intensities.
    spacing: array (n, )
        the image spacing.
    shape: array (x, y, z)
        the input image shape.
    size: array (n, )
        the image size.
    half_patch_size: array (x, y, z)
        the half size of the denoising patches in voxel.
    full_patch_size: array (x, y, z)
        the full size of the denoising patches in voxel: 2 * half_patch_size + 1.
    """

    # Global parameters
    valid_blockwise_strategies = ["pointwise", "blockwise", "fastblockwise"]
    valid_cental_point_strategies = ["add", "remove", "weight"]

    def __init__(self, to_denoise_array, spacing, mask_array=None,
                 half_patch_size=1, half_spatial_bandwidth=5,
                 central_point_strategy="weight", blockwise_strategy="blockwise",
                 lower_mean_threshold=0.95, lower_variance_threshold=0.5,
                 beta=1, use_optimized_strategy=True, use_cython=True,
                 nb_of_threads=1):
        """ Initialize the non-local denoising class.
    
        Parameters
        ----------
        array_to_denoise: array
            an input n dimensional array to process containing the image
            intensities.
        spacing: n-uplet
            the image spacing in voxel.
        half_patch_size: int
            the patch size in voxel.
        central_point_strategy: str (default 'weight')
            the way the central patch is considered, one of 'add', 'remove' or
            'weight'.
        blockwise_strategy: str (default 'blockwise')
            the blockwise denoising strategy, one of 'pointwise', 'blockwise' or
            'fastblockwise'.
        beta: float
            smoothing parameter.
        use_cython: bool (default True)
            the cython to speed up the denoised patch creation.
        nb_of_threads: int (default 1)
            if cython code, defines the number of threads to use.
        """
      
        # Class parameters
        self.to_denoise_array = numpy.cast[numpy.single](to_denoise_array)
        self.spacing = numpy .asarray(spacing)
        self.mask_array = mask_array
        self.mean_array = None
        self.variance_array = None
        self.central_point_strategy = central_point_strategy
        self.blockwise_strategy = blockwise_strategy
        self.use_optimized_strategy = use_optimized_strategy
        self.lower_mean_threshold = lower_mean_threshold
        self.lower_variance_threshold = lower_variance_threshold
        self.use_cython = use_cython
        self.nb_of_threads = nb_of_threads

        # Intern parameters
        self.shape = self.to_denoise_array.shape
        self.size = self.to_denoise_array.size
        
        # Check that the mask array has a valid shape
        if self.mask_array is not None:
            if self.shape != self.mask_array.shape:
                raise ValueError("Input mask array has invalid shape.")
        else:
            self.mask_array = numpy.ones(self.shape, dtype=numpy.int)
        remaining_voxels = len(numpy.where(self.mask_array > 0)[0])
        logger.info("Remaining voxels to process '%s' percent.",
                    int(remaining_voxels / self.size * 100.))

        # Check that a valid denoising strategy has been selected.
        if self.blockwise_strategy not in self.valid_blockwise_strategies:
            raise ValueError(
                "Wrong denoising strategy '{0}'. Valid options are '{1}'.".format(
                    self.blockwise_strategy, self.valid_blockwise_strategies))

        # Check that a valid central point strategy has been selected.
        if self.central_point_strategy not in self.valid_cental_point_strategies:
            raise ValueError(
                "Wrong central point strategy '{0}'. Valid options are '{1}'.".format(
                    self.central_point_strategy, self.valid_cental_point_strategies))
        
        # Intern parameters
        # > patch size
        self.half_patch_size, self.full_patch_size = normalize_patch_size(
            half_patch_size, self.spacing)

        # > spatial bandwidth: equivalent to the size of the volume search area
        # in non-local means
        (self.half_spatial_bandwidth, 
         self.full_spatial_bandwidth) = normalize_patch_size(
            half_spatial_bandwidth, self.spacing)

        # > compute mean and variance images
        if self.use_optimized_strategy:             
            self._compute_mean_and_variance()

        # > smooth parameter      
        self.range_bandwidth = self._compute_range_bandwidth(beta)

    ###########################################################################
    # Public methods
    ###########################################################################

    def denoise(self):
        """ Method that computes the denoised image using NLM algorithm.

        Returns
        -------
        denoise_array: array
            the denoised image.
        """
        # Information message
        logger.info("Compute the denoised image using NLM algorithm.")

        # Allocate the denoised output image and the weight image
        denoise_array = numpy.zeros(self.shape, dtype=numpy.single)
        weights = numpy.zeros(self.shape, dtype=numpy.single)

        # Go through all the voxels and compute the denoise image
        logger.info("Running denoising...")
        for vector_index in range(self.to_denoise_array.size):

            # Convert vector index to array index
            index = vector_to_array_index(vector_index, self.to_denoise_array)

            # Falst blockwise non overlap speed up
            if (self.blockwise_strategy != "fastblockwise" or 
                (numpy.mod(index, self.half_patch_size + 1) == 0).all()):

                # Check if have to compute this voxel
                if self.mask_array[tuple(index)] > 0:

                    # Compute the denoised patch
                    patch, weight = self._get_denoised_patch(index)

                    # Apply the denoised patch
                    self._apply_patch(index, patch, denoise_array, weights,
                                      weight=1.) 

        # Apply the weightings to the denoised image
        weights[numpy.where(weights == 0)] = 1.
        denoise_array /= weights

        return denoise_array
    
    ###########################################################################
    # Private methods
    ###########################################################################

    def _get_search_elements(self, index):
        """ Method that compute the neighbor search indices around the
        current location.

        Parameters
        ----------
        index: array
            the current location.

        Returns 
        -------
        search_elements: list of array
            the list with all the neighbor indices.
        """
        # Create an appropriate search region around the current voxel index
        # neglecting pixels outside the boundaries
        lower_bound = index - self.half_spatial_bandwidth
        upper_bound = index + self.half_spatial_bandwidth

        # Get the search region
        search_elements = []
        for x in range(lower_bound[0], upper_bound[0] + 1):
            for y in range(lower_bound[1], upper_bound[1] + 1):
                for z in range(lower_bound[2], upper_bound[2] + 1):

                    # Check we are within the image boundaries
                    neighbor_index = numpy.asarray([x, y, z])
    
                    if ((neighbor_index >= 0).all() and
                        (neighbor_index < self.shape).all() and
                        not (neighbor_index == index).all()):

                        search_elements.append(neighbor_index)

        return search_elements
        
    def _check_speed(self, index1, index2):
        """ Check the lower mean and varaince thresholds.

        Parameters
        ----------
        index1, index2: tuple
            two image index to test for processing speed.

        Returns
        -------
        is_valid: bool
            the two index can be computed in the optimized settings.
        """
        # Compute the likelihood between the two means
        if self.mean[index2] == 0:
            if self.mean[index1] == 0:
                mean_likelihood = 1
            else:
                mean_likelihood = 0
        else:
            mean_likelihood = self.mean[index1] / self.mean[index2]

        # Check speed
        if (mean_likelihood < self.lower_mean_threshold or 
            mean_likelihood > 1. / self.lower_mean_threshold):
            
            return False

        # Compute the likelihood between the two variances
        if self.variance[index2] == 0:
            if self.variance[index1] == 0:
                variance_likelihood = 1
            else:
                variance_likelihood = 0
        else:
            variance_likelihood = self.variance[index1] / self.variance[index2]

        # Check speed
        if (variance_likelihood < self.lower_variance_threshold or 
            variance_likelihood > 1. / self.lower_variance_threshold):
            
            return False

        return True

    def _get_denoised_patch(self, index):
        """ Method that compute the denoised patch at a specific location.

        Parameters
        ----------
        index: array
            the patch central location.

        Returns
        -------
        patch: aray
            the denoised patch.
        weight: float
            the patch power (ie., the sum of all the weights associated to
            the neighbor patches).
        """
        # Get patch around the current location.
        central_patch = get_patch(
            index, self.to_denoise_array, self.full_patch_size)

        if not self.use_cython:
            # Intern parameters
            # > maximum weight of patches
            wmax = 0
            # > sum of weights: used for normalization
            wsum = 0

            # Allocate the patch result
            patch = numpy.zeros(self.full_patch_size, dtype=numpy.single)

            # Compute the search region
            search_elements = self._get_search_elements(index)

            # Go through all the search indices
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
                    patch += neighbor_patch * weight
        else:
            # Use compiled code to do the same steps as the upper python version
            patch, wsum, wmax = get_average_patch(
                self.to_denoise_array, index, self.half_spatial_bandwidth,
                self.half_patch_size, self.full_patch_size, self.range_bandwidth,
                nb_of_threads=self.nb_of_threads)

        # Deal with the central patch based on the user parameters
        # > add the central patch
        if self.central_point_strategy == "add":
            patch += central_patch
            wsum += 1.
        # > remove the central patch
        elif self.central_point_strategy == "remove":
            pass
        # > use a weighted central patch
        elif self.central_point_strategy == "weight":
            patch += (1. - wmax) * central_patch
            wsum += (1. - wmax)
        # > raise value error
        else:
            raise ValueError("Unexpected central point strategy '{0}'.".format(
                self.central_point_strategy))

        # Normalize the denoised patch if the total weight is big enough
        if wsum > 0.0001:
            patch /= wsum
        # Otherwise can't denoise properly, copy the central patch to the 
        # denoise patch
        else:
            patch = central_patch

        return patch, wsum        

    def _apply_patch(self, index, patch, denoise_array, weights, weight=1.):
        """ Method that apply the selected denoising patch strategy.

        Parameters
        ----------
        index: array
            the current voxel index that is denoised.
        patch: array
            the denoised patch around the current voxel.
        denoise_array: array
            the final denoised image.
        weights: array
            the weights that will be applied on the denoised image to get the
            final result
        weight: float (default 1)
            the weight associated to the current patch.
        """
        # Only consider the central patch voxel, do not apply any weighting
        # strategy
        if self.blockwise_strategy == "pointwise":
            denoise_array[tuple(index)] = patch[tuple(self.half_patch_size)]
            weights[tuple(index)] = weight
            
        # Consider all the patch spatial extension, and apply the patch
        # weight
        elif self.blockwise_strategy in ["blockwise", "fastblockwise"]:

            # Create an appropriate patch region around the current voxel index
            # neglecting pixels close to the boundaries
            lower_bound = index - self.half_patch_size
            upper_bound = index + self.half_patch_size

            # Check that the patch is inside the image
            if (lower_bound < 0).any() or (upper_bound >= self.shape).any():
                logger.debug("Reach the border when creating patch at position "
                             "'%s'.", index)
                return
            
            # Add the patch values and the weight factor
            denoise_array[lower_bound[0]: upper_bound[0] + 1,
                          lower_bound[1]: upper_bound[1] + 1,
                          lower_bound[2]: upper_bound[2] + 1] += patch
            weights[lower_bound[0]: upper_bound[0] + 1,
                    lower_bound[1]: upper_bound[1] + 1,
                    lower_bound[2]: upper_bound[2] + 1] += weight
           
        # Raise a value error
        else:
            raise ValueError("Unexpected denoising strategy '{0}'.".format(
                self.blockwise_strategy))

    def _compute_range_bandwidth(self, beta):
        """ Compute the range bandwidth.

        Parameters
        ----------
        beta: float
            smoothing parameter.

        Returns
        -------
        range_bandwidth: float
            the global bandwidth range.
        """
        # Information message
        logger.info("Computing the range bandwidth corresponding to the "
                    "smooth parameter.")

        # Build the structural element
        struct_element = numpy.zeros((3, 3, 3), dtype=numpy.single)
        struct_element[:, 1, 1] = -1. / 6.
        struct_element[1, :, 1] = -1. / 6.
        struct_element[1, 1, :] = -1. / 6.
        struct_element[1, 1, 1] = 1.

        # Neglect the border during the convolution
        ei = numpy.sqrt(6. / 7.) * scipy.signal.fftconvolve(
            self.to_denoise_array, struct_element, mode="same")
        ei = ei[numpy.where(self.mask_array > 0)]

        # Estimate sigma with MAD
        pos_ei = numpy.sort(ei[numpy.where(ei > 0)])
        median_value = pos_ei[pos_ei.size / 2]
        pos_ei = numpy.sort(numpy.abs(pos_ei - median_value))
        sigma = 1.4826 * pos_ei[pos_ei.size / 2]
        temp = 2. * self.half_patch_size + 1.
        range_bandwidth = 2. * beta * sigma**2 * temp[0] * temp[1] * temp[2]

        # Information message
        logger.info("The global standard deviation is '%s'. The estimation "
                    "of the parameter was performed on '%s' points.",
                    sigma, ei.size)

        return range_bandwidth

    def _compute_mean_and_variance(self):
        """ Compute the mean and the variance of the image to denoise.
        """
        # Information message
        logger.info("Optimized mode, compute mean and variance image.") 

        # Allocate the two arrays
        self.mean = numpy.zeros(self.shape, dtype=numpy.single)
        self.variance  = numpy.zeros(self.shape, dtype=numpy.single)

        # Go through all the voxels
        for vector_index in range(self.to_denoise_array.size):

            # Convert vector index to array index
            index = vector_to_array_index(vector_index, self.to_denoise_array)

            # Check if have to compute this voxel
            if self.mask_array[tuple(index)] > 0:

                # Get the surrounding patch
                patch = get_patch(
                    index, self.to_denoise_array, self.full_patch_size)

                # Compute the local mean and variance
                local_mean = numpy.mean(patch)
                local_variance = numpy.mean(patch**2) - local_mean**2

                # Store the computed values
                self.mean[tuple(index)] = local_mean
                self.variance[tuple(index)] = local_variance


if __name__ == "__main__":

    # IO import
    import nibabel

    # System import
    import datetime


    # Synthetic simulation
    to_denoise_array = numpy.random.randn(10, 10, 10)
    to_denoise_array += 10
    to_denoise_array[5:, ...] -= 5
    to_denoise_array = numpy.cast[numpy.single](to_denoise_array)
    image = nibabel.Nifti1Image(data=to_denoise_array, affine=numpy.eye(4))
    nibabel.save(image, "/home/grigis/tmp/noise.nii.gz")
    nlm_filter = NLMDenoising(to_denoise_array, (1, 1, 1),
                              blockwise_strategy="blockwise",
                              half_spatial_bandwidth=3)
    denoise_array = nlm_filter.denoise()
    image = nibabel.Nifti1Image(data=denoise_array, affine=numpy.eye(4))
    nibabel.save(image, "/home/grigis/tmp/nlm.nii.gz")

    # Real data: this step is long, start a timer
    start_time = datetime.datetime.now()
    to_denoise_image = nibabel.load("/home/grigis/tmp/data.nii.gz")
    array = to_denoise_image.get_data()[..., 0]
    spacing = to_denoise_image.get_header().get_zooms()[:3]
    affine = to_denoise_image.get_affine()
    image = nibabel.Nifti1Image(data=array, affine=affine)
    nibabel.save(image, "/home/grigis/tmp/data_noise.nii.gz")
    nlm_filter = NLMDenoising(array, spacing,
                              blockwise_strategy="blockwise",
                              half_spatial_bandwidth=5,
                              use_optimized_strategy=False)
    denoise_array = nlm_filter.denoise()
    image = nibabel.Nifti1Image(data=denoise_array, affine=affine)
    nibabel.save(image, "/home/grigis/tmp/data_nlm.nii.gz")

    # Stop the timer
    delta_time = datetime.datetime.now() - start_time
    print "\nDone in {0} seconds.".format(delta_time)

    


