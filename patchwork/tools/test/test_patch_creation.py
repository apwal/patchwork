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
from patchwork.tools import get_patch
from patchwork.tools import vector_to_array_index


class TestPatchCreation(unittest.TestCase):
    """ Class to test the creation of a patch.
    """
    def setUp(self):
        """ Test settings.
        """
        self.shape = (100, 100, 50)
        self.array = numpy.ones(self.shape, dtype=numpy.int)
        self.locations = [
            numpy.array([0, 0, 0]),
            numpy.array([3, 5, 2])
        ]
        self.full_patch_size = numpy.array([3, 3, 3], dtype=numpy.int)
        self.results = [
            numpy.zeros(self.full_patch_size, dtype=numpy.int),
            numpy.ones(self.full_patch_size, dtype=numpy.int)
        ]            

    def test_patch_creation(self):
        """ Test the patch creation at different locations.
        """
        for index, target in zip(self.locations, self.results):
            patch = get_patch(index, self.array, self.full_patch_size)
            self.assertTrue(numpy.allclose(patch, target))

    def test_speed(self):
        """ Test the patch creation speed.
        """
        for vector_index in range(self.array.size):
            index = vector_to_array_index(vector_index, self.array)
            patch = get_patch(index, self.array, self.full_patch_size)

def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatchCreation)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
