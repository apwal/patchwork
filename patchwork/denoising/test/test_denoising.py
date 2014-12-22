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


class TestDenoising(unittest.TestCase):
    """ Class to test the the patch denoising methods.
    """
    def setUp(self):
        """ Test settings.
        """
        self.to_denoise_array = numpy.random.randn(10, 10, 10)
        self.to_denoise_array[..., :5] = 5
        self.kwargs = {
            "spacing": (1, 1, 1),
            "mask_array": None,
            "half_patch_size": 1,
            "half_spatial_bandwidth": 5,
            "padding": -1,
            "central_point_strategy": "weight",
            "blockwise_strategy": "blockwise",
            "lower_mean_threshold": 0.95,
            "lower_variance_threshold": 0.5,
            "beta": 1,
            "use_optimized_strategy": True
        }

    def test_denoising(self):
        """ Test the nlm denoising.
        """
        nlm_filter = NLMDenoising(
            self.to_denoise_array, **self.kwargs)
        denoise_array = nlm_filter.denoise()
    


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDenoising)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
