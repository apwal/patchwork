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
from patchwork.tools import normalize_patch_size


class TestPatchNormalization(unittest.TestCase):
    """ Class to test the the patch size normalization.
    """
    def setUp(self):
        """ Test settings.
        """
        self.spacings = [[1, 1, 1, 1], [3, 1, 2, 3], [2, 2, 2, 4]]
        self.patch_sizes = [numpy.array([1, 1, 1]), numpy.array([3, 1, 1]),
                            numpy.array([2, 2, 1])]
        
    def test_patch_size(self):
        """ Test the patch size normalization according to the image anisotropy.
        """
        for spacing, patch_size in zip(self.spacings, self.patch_sizes):
            half_patch_size, full_patch_size = normalize_patch_size(
                spacing[0], spacing[1:])
            self.assertEqual(
                half_patch_size.tolist(), patch_size.tolist())
            self.assertEqual(
                full_patch_size.tolist(), (2 * patch_size + 1).tolist()) 


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatchNormalization)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
