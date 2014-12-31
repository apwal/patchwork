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
from patchwork.tools import patch_mean_variance
from patchwork.tools import c_patch_mean_variance


class TestPatchMeanVariance(unittest.TestCase):
    """ Class to test the creation of a patch.
    """
    def setUp(self):
        """ Test settings.
        """
        shape = (100, 100, 50)
        self.array = numpy.random.randn(shape[0], shape[1], shape[2])
        self.array += 10
        self.array[5:, ...] -= 5
        self.array = numpy.cast[numpy.single](self.array)
        self.mask = numpy.zeros(shape, dtype=numpy.single)
        self.patch_shape = numpy.array([3, 3, 3], dtype=numpy.int)       

    def test_mean_varaince(self):
        """ Test the patch mean and variance computation.
        """
        mean, variance = patch_mean_variance(
            self.array, self.mask, self.patch_shape)
        cmean, cvariance = c_patch_mean_variance(
            self.array, self.mask, self.patch_shape)
        self.assertTrue(numpy.allclose(mean, cmean))
        self.assertTrue(numpy.allclose(variance, cvariance))


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatchMeanVariance)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
