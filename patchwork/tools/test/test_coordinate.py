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
from patchwork.tools import vector_to_array_index, array_to_vector_index


class TestCoordinate(unittest.TestCase):
    """ Class to test the coordinate system tools.
    """
    def setUp(self):
        """ Test settings.
        """
        self.shape = (100, 100, 50)
        self.array = numpy.zeros(self.shape, dtype=numpy.int)
        self.positions = [
            (0, 0, 1), (0, 1, 0), (1, 0, 0),
            (24, 46, 3), (99, 99, 49)]
        self.values = range(1, len(self.positions) + 1)
        
    def test_vector_to_array(self):
        """ Test the vector to array coordinate system.
        """
        for pos, val in zip(self.positions, self.values):
            self.array[pos] = val
            vector_index = array_to_vector_index(pos, self.array)
            self.assertEqual(self.array.flatten()[vector_index], val)
            array_index = vector_to_array_index(vector_index, self.array)
            self.assertEqual(tuple(array_index), pos)


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoordinate)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
