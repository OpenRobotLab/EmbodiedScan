# Copyright (c) OpenRobotLab. All rights reserved.
import numpy as np
import torch
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

from ..points.base_points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_euler


class EulerInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes with 1-D orientation represented by three Euler angles.

    See https://en.wikipedia.org/wiki/Euler_angles for
        regarding the definition of Euler angles.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, alpha, beta, gamma).
    """

    def __init__(self, tensor, box_dim=9, origin=(0.5, 0.5, 0.5)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32,
                                                     device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # (0, 0, 0) as a fake euler angle.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 3)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 3
        elif tensor.shape[-1] == 7:
            assert box_dim == 7
            fake_euler = tensor.new_zeros(tensor.shape[0], 2)
            tensor = torch.cat((tensor, fake_euler), dim=-1)
            self.box_dim = box_dim + 2
        else:
            assert tensor.shape[-1] == 9
            self.box_dim = box_dim
        self.tensor = tensor.clone()

        self.origin = origin
        if origin != (0.5, 0.5, 0.5):
            dst = self.tensor.new_tensor((0.5, 0.5, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def get_corners(self, tensor1):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        if tensor1.numel() == 0:
            return torch.empty([0, 8, 3], device=tensor1.device)

        dims = tensor1[:, 3:6]
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3),
                     axis=1)).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin
        assert self.origin == (0.5, 0.5, 0.5), \
            'self.origin != (0.5, 0.5, 0.5) needs to be checked!'
        corners_norm = corners_norm - dims.new_tensor(self.origin)
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate
        corners = rotation_3d_in_euler(corners, tensor1[:, 6:])

        corners += tensor1[:, :3].view(-1, 1, 3)
        return corners

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou', eps=1e-4):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`EulerInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`EulerInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.
            eps (bool): Epsilon. Defaults to 1e-4.

        Returns:
            torch.Tensor: Calculated 3D overlaps of the boxes.
        """
        assert isinstance(boxes1, EulerInstance3DBoxes)
        assert isinstance(boxes2, EulerInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        corners1 = boxes1.corners
        corners2 = boxes2.corners
        _, iou3d = box3d_overlap(corners1, corners2, eps=eps)
        return iou3d

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3),
                     axis=1)).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin
        assert self.origin == (0.5, 0.5, 0.5), \
            'self.origin != (0.5, 0.5, 0.5) needs to be checked!'
        corners_norm = corners_norm - dims.new_tensor(self.origin)
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate
        corners = rotation_3d_in_euler(corners, self.tensor[:, 6:])

        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def transform(self, matrix):
        if self.tensor.shape[0] == 0:
            return
        if not isinstance(matrix, torch.Tensor):
            matrix = self.tensor.new_tensor(matrix)
        points = self.tensor[:, :3]
        constant = points.new_ones(points.shape[0], 1)
        points_extend = torch.concat([points, constant], dim=-1)
        points_trans = torch.matmul(points_extend, matrix.transpose(-2,
                                                                    -1))[:, :3]

        size = self.tensor[:, 3:6]

        # angle_delta = matrix_to_euler_angles(matrix[:3,:3], 'ZXY')
        # angle = self.tensor[:,6:] + angle_delta
        ori_matrix = euler_angles_to_matrix(self.tensor[:, 6:], 'ZXY')
        rot_matrix = matrix[:3, :3].expand_as(ori_matrix)
        final = torch.bmm(rot_matrix, ori_matrix)
        angle = matrix_to_euler_angles(final, 'ZXY')

        self.tensor = torch.cat([points_trans, size, angle], dim=-1)
    
    def scale(self, scale_factor: float) -> None:
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns
                None, otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)

        if angle.numel() == 1:  # only given yaw angle for rotation
            angle = self.tensor.new_tensor([angle, 0., 0.])
            rot_matrix = euler_angles_to_matrix(angle, 'ZXY')
        elif angle.numel() == 3:
            rot_matrix = euler_angles_to_matrix(angle, 'ZXY')
        elif angle.shape == torch.Size([3, 3]):
            rot_matrix = angle
        else:
            raise NotImplementedError

        rot_mat_T = rot_matrix.T
        transform_matrix = torch.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        self.transform(transform_matrix)

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T
        else:
            return rot_mat_T

    def flip(self, direction='X'):
        """Flip the boxes along the corresponding axis.

        Args:
            direction (str, optional): Flip axis. Defaults to 'X'.
        """
        assert direction in ['X', 'Y', 'Z']
        if direction == 'X':
            self.tensor[:, 0] = -self.tensor[:, 0]
            self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
            self.tensor[:, 8] = -self.tensor[:, 8]
        elif direction == 'Y':
            self.tensor[:, 1] = -self.tensor[:, 1]
            self.tensor[:, 6] = -self.tensor[:, 6]
            self.tensor[:, 7] = -self.tensor[:, 7] + np.pi
        elif direction == 'Z':
            self.tensor[:, 2] = -self.tensor[:, 2]
            self.tensor[:, 7] = -self.tensor[:, 7]
            self.tensor[:, 8] = -self.tensor[:, 8] + np.pi
