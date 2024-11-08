# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .test import TESTDataset as test
from .unreal import UNRLDataset as unreal
from .vist import VISTDataset as vist
from .unreal_depth import UNRLDepthDataset as unrealD
from .unreal_TC import UNRLTCDataset as unrealTC
from .unreal_sigma import UNRLSDataset as unrealS
