# Copyright (c) OpenRobotLab. All rights reserved.
import numpy as np
import torch
from mmcv.ops import points_in_boxes_all, points_in_boxes_part

from ..points.base_points import BasePoints
from .euler_box3d import EulerInstance3DBoxes


class EulerDepthInstance3DBoxes(EulerInstance3DBoxes):
    """3D boxes of instances in Depth coordinates.

    We keep the "Depth" coordinate system definition in MMDet3D just for
    clarification of the points coordinates and the flipping augmentation.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (alpha=0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (alpha=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, alpha, beta, gamma).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self,
                 tensor,
                 box_dim=9,
                 with_yaw=True,
                 origin=(0.5, 0.5, 0.5)):
        super().__init__(tensor, box_dim, origin)
        self.with_yaw = with_yaw

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction.

        In Depth coordinates, it flips x (horizontal) or y (vertical) axis.

        Args:
            bev_direction (str, optional): Flip direction
                (horizontal or vertical). Defaults to 'horizontal'.
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            super().flip(direction='X')
        elif bev_direction == 'vertical':
            super().flip(direction='Y')

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 1] = -points[:, 1]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`DepthInstance3DBoxes`:
                The converted box of the same type in the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        assert dst == Box3DMode.EULER_DEPTH
        return self

    def points_in_boxes_part(self, points, boxes_override=None):
        """Find the box in which each point is.

        Args:
            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (torch.Tensor, optional): Boxes to override
                `self.tensor`. Defaults to None.

        Returns:
            torch.Tensor: The index of the first box that each point
                is in, in shape (M, ). Default value is -1
                (if the point is not enclosed by any box).

        Note:
            If a point is enclosed by multiple boxes, the index of the
            first box will be returned.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        if points.dim() == 2:
            points = points.unsqueeze(0)
        # TODO: take euler angles into consideration
        aligned_boxes = boxes[..., :7].clone()
        aligned_boxes[..., 6] = 0
        box_idx = points_in_boxes_part(
            points,
            aligned_boxes.unsqueeze(0).to(points.device)).squeeze(0)
        return box_idx

    def points_in_boxes_all(self, points, boxes_override=None):
        """Find all boxes in which each point is.

        Args:
            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (torch.Tensor, optional): Boxes to override
                `self.tensor`. Defaults to None.

        Returns:
            torch.Tensor: A tensor indicating whether a point is in a box,
                in shape (M, T). T is the number of boxes. Denote this
                tensor as A, if the m^th point is in the t^th box, then
                `A[m, t] == 1`, elsewise `A[m, t] == 0`.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1

        boxes = boxes.to(points_clone.device).unsqueeze(0)
        # TODO: take euler angles into consideration
        aligned_boxes = boxes[..., :7].clone()
        aligned_boxes[..., 6] = 0
        box_idxs_of_pts = points_in_boxes_all(points_clone, aligned_boxes)

        return box_idxs_of_pts.squeeze(0)
