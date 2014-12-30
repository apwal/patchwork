#! /usr/bin/env python
################################################################################
# PATCHWORK - Copyright (C) 2014
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
################################################################################

# System import
import os
import numpy
import unittest

# Patchwork import
from patchwork.denoising import NLMDenoising
from patchwork.tools import get_patch as py_get_patch
from patchwork.denoising.nlm_core import get_patch as c_get_patch


class TestDenoising(unittest.TestCase):
    """ Class to test the the patch denoising methods.
    """
    def setUp(self):
        """ Test settings.
        """
        self.to_denoise_array = numpy.random.randn(10, 10, 10)
        self.to_denoise_array += 10
        self.to_denoise_array[5:, ...] -= 5
        self.to_denoise_array = numpy.cast[numpy.single](self.to_denoise_array)
        self.kwargs = {
            "spacing": (1, 1, 1),
            "mask_array": None,
            "half_patch_size": 1,
            "half_spatial_bandwidth": 5,
            "padding": -1,
            "central_point_strategy": "weight",
            "blockwise_strategy": "fastblockwise",
            "lower_mean_threshold": 0.95,
            "lower_variance_threshold": 0.5,
            "beta": 1,
            "use_optimized_strategy": False,
            "use_cython": True
        }

    def test_patch_denoising(self):
        """ Test the denoised patch creation.
        """
        # Denoising parameters
        self.kwargs["half_spatial_bandwidth"] = 3
        self.kwargs["use_optimized_strategy"] = False
        self.kwargs["central_point_strategy"] = "remove"
        indices = [
            numpy.array((8, 8, 2), dtype=numpy.int),
            numpy.array((5, 5, 5), dtype=numpy.int)
        ]

        # Test all indices
        for index in indices:

            # Without C speed up
            self.kwargs["use_cython"] = False
            nlm_filter = NLMDenoising(
                self.to_denoise_array, **self.kwargs)
            py_patch, py_weight = nlm_filter._get_denoised_patch(index)

            # With C speed up
            self.kwargs["use_cython"] = True
            nlm_filter = NLMDenoising(
                self.to_denoise_array, **self.kwargs)
            c_patch, c_weight = nlm_filter._get_denoised_patch(index)

            # Test equality
            self.assertTrue(numpy.allclose(py_patch, c_patch))
            self.assertTrue(numpy.allclose(py_weight, c_weight)) 

    def test_patch_creation(self):
        """ Test the patch creation.
        """
        index = numpy.array((4, 5, 2), dtype=numpy.int)
        patch_shape = numpy.array((3, 3, 3), dtype=numpy.int)
        py_patch = py_get_patch(index, self.to_denoise_array, patch_shape)
        c_patch = c_get_patch(index, self.to_denoise_array, patch_shape)
        self.assertTrue(numpy.allclose(py_patch, c_patch)) 

    def test_denoising(self):
        """ Test the nlm denoising.
        """
        # Denoising parameters
        self.kwargs["half_spatial_bandwidth"] = 5
        self.kwargs["use_optimized_strategy"] = True
        self.kwargs["central_point_strategy"] = "add"
        self.kwargs["use_cython"] = True

        # Test all denoising strategies
        for strategy in ["pointwise", "fastblockwise", "blockwise"]:
            self.kwargs["blockwise_strategy"] = strategy
            nlm_filter = NLMDenoising(
                self.to_denoise_array, **self.kwargs)
            denoise_array = nlm_filter.denoise()

    def test_fast_denoising(self):
        """ Test the nlm denoising: python vs cython.
        """
        # Denoising parameters
        self.kwargs["half_spatial_bandwidth"] = 3
        self.kwargs["blockwise_strategy"] = "blockwise"
        self.kwargs["use_optimized_strategy"] = False
        self.kwargs["central_point_strategy"] = "remove"

        # Without C speed up
        self.kwargs["use_cython"] = False
        nlm_filter = NLMDenoising(
            self.to_denoise_array, **self.kwargs)
        py_denoise_array = nlm_filter.denoise()

        # With C speed up
        self.kwargs["use_cython"] = True
        nlm_filter = NLMDenoising(
            self.to_denoise_array, **self.kwargs)
        c_denoise_array = nlm_filter.denoise()

        # Test equality
        self.assertTrue(
            numpy.allclose(py_denoise_array, c_denoise_array))


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDenoising)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
