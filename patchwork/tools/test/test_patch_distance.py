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
from patchwork.tools import patch_distance


class TestPatchDistance(unittest.TestCase):
    """ Class to test the metric between two patches.
    """
    def setUp(self):
        """ Test settings.
        """
        self.patches1 = [
            numpy.zeros((3, 3, 3), dtype=numpy.single),
            numpy.ones((3, 3, 3), dtype=numpy.single),
            numpy.zeros((3, 3, 3), dtype=numpy.single)]
        self.patches2 = [
            numpy.zeros((3, 3, 3), dtype=numpy.single),
            numpy.ones((3, 3, 3), dtype=numpy.single),
            numpy.ones((3, 3, 3), dtype=numpy.single)] 
        self.metrics = [0., 0., 27.]          

    def test_patch_distance(self):
        """ Test the squared euclidean distance between patches.
        """
        for patch1, patch2, metric in zip(self.patches1, self.patches2,
                                          self.metrics):
            dist = patch_distance(patch1, patch2)
            self.assertEqual(dist, metric)

def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatchDistance)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
