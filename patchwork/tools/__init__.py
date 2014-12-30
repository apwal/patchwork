#! /usr/bin/env python
################################################################################
# PATCHWORK - Copyright (C) 2014
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
################################################################################

from .patch_creator import patch_distance
from .patch_creator import get_patch
from .patch_creator import get_patch_elements
from .patch_creator import normalize_patch_size

from .patch_core import get_patch as c_get_patch

from .coordinate import vector_to_array_index
from .coordinate import array_to_vector_index

__all__ = ["get_patch", "c_get_patch", "patch_distance", "get_patch_elements",
           "normalize_patch_size", "vector_to_array_index",
           "array_to_vector_index"]
