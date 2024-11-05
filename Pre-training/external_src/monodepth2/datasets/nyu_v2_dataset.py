# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class NYUv2BaseDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NYUv2BaseDataset, self).__init__(*args, **kwargs)

        fx_rgb = 518.85790117450188
        fy_rgb = 519.46961112127485
        cx_rgb = 325.58244941119034
        cy_rgb = 253.73616633400465
        intrinsic_matrix = np.array([
            [fx_rgb,   0.0,     cx_rgb,  0.0],
            [0.0,      fy_rgb,  cy_rgb,  0.0],
            [0.0,      0.0,     1.0,     0.0],
            [0.0,      0.0,     0.0,     1.0]], dtype=np.float32)

        d_height = 480 - 416
        d_width = 640 - 576

        y_start = d_height // 2
        x_start = d_width // 2

        intrinsic_matrix = intrinsic_matrix + [[0.0, 0.0, -x_start, 0.0],
                                               [0.0, 0.0, -y_start, 0.0],
                                               [0.0, 0.0, 0.0,      0.0],
                                               [0.0, 0.0, 0.0,      0.0]]

        self.K = intrinsic_matrix.astype(np.float32)

        self.full_res_shape = (576, 416)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return None

    def get_color(self, path, do_flip):
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        color1, color0, color2 = np.split(np.asarray(color), indices_or_sections=3, axis=1)

        color0 = pil.fromarray(color0)
        color1 = pil.fromarray(color1)
        color2 = pil.fromarray(color2)

        return color0, color1, color2


class NYUv2Dataset(NYUv2BaseDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYUv2Dataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        return None

    def get_depth(self, folder, frame_index, side, do_flip):
        return None
