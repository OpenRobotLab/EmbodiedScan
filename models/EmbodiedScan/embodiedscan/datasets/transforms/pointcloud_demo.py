import json
import os

import numpy as np
import torch
from embodiedscan.registry import TRANSFORMS
from embodiedscan.structures.points import DepthPoints, get_points_type
from lry_utils.utils_read import NUM2RAW_3RSCAN, to_sample_idx, to_scene_id
from mmcv.transforms import BaseTransform, Compose


@TRANSFORMS.register_module()
class PointCloudPipelineDemo(BaseTransform):
    """Multiview data processing pipeline.

    The transform steps are as follows:

        1. Select frames.
        2. Re-ororganize the selected data structure.
        3. Apply transforms for each selected frame.
        4. Concatenate data to form a batch.

    Args:
        transforms (list[dict | callable]):
            The transforms to be applied to each select frame.
        n_images (int): Number of frames selected per scene.
        ordered (bool): Whether to put these frames in order.
            Defaults to False.
    """

    def __init__(self, ordered=False, keep_rgb=True):
        super().__init__()
        self.ordered = ordered
        self.keep_rgb = keep_rgb

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        scene_id = results['scan_id']
        assert scene_id in ['office', 'restroom', 'restroom2'], scene_id
        pcd_path = f'/mnt/petrelfs/lvruiyuan/repos/EmbodiedScan/data/open_scan/{scene_id}.npy'
        pc_with_color = np.load(pcd_path)
        pc, color = pc_with_color[:, :3], pc_with_color[:, 3:]
        color = color / 255
        if self.keep_rgb:
            points = np.concatenate([pc, color], axis=-1)
            points = DepthPoints(points,
                                 points_dim=6,
                                 attribute_dims=dict(color=[
                                     points.shape[1] - 3,
                                     points.shape[1] - 2,
                                     points.shape[1] - 1,
                                 ]))
        else:
            points = DepthPoints(pc)
        _results = {'points': points}
        results.update(_results)
        return results
