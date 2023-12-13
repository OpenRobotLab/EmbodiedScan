# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet3d.structures.points import BasePoints
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_euler


class EulerInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in Depth coordinates.

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

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

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
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 3)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 3
            self.with_yaw = True  # TODO
        elif tensor.shape[-1] == 7:
            assert box_dim == 7
            fake_euler = tensor.new_zeros(tensor.shape[0], 2)
            tensor = torch.cat((tensor, fake_euler), dim=-1)
            self.box_dim = box_dim + 2
            self.with_yaw = True
        else:
            assert tensor.shape[-1] == 9
            self.box_dim = box_dim
            self.with_yaw = True  # TODO
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
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

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
        _, iou3d = box3d_overlap(corners1, corners2, eps=1e-4)
        return iou3d

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        raise NotImplementedError('Not support')

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
            angle = self.tensor.new_tensor([0., 0., angle])
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

    def flip(self, direction='X', points=None):
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

        if points is not None:
            # assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if direction == 'X':
                    points[:, 0] = -points[:, 0]
                elif direction == 'Y':
                    points[:, 1] = -points[:, 1]
                elif direction == 'Z':
                    points[:, 2] = -points[:, 2]
            else:
                points.flip(direction)
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

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`DepthInstance3DBoxes`: Enlarged boxes.
        """
        raise NotImplementedError('enlarged box')
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def get_surface_line_center(self):
        """Compute surface and line center of bounding boxes.

        Returns:
            torch.Tensor: Surface and line center of bounding boxes.
        """
        raise NotImplementedError('surface line center')
        obj_size = self.dims
        center = self.gravity_center.view(-1, 1, 3)
        batch_size = center.shape[0]

        rot_sin = torch.sin(-self.yaw)
        rot_cos = torch.cos(-self.yaw)
        rot_mat_T = self.yaw.new_zeros(tuple(list(self.yaw.shape) + [3, 3]))
        rot_mat_T[..., 0, 0] = rot_cos
        rot_mat_T[..., 0, 1] = -rot_sin
        rot_mat_T[..., 1, 0] = rot_sin
        rot_mat_T[..., 1, 1] = rot_cos
        rot_mat_T[..., 2, 2] = 1

        # Get the object surface center
        offset = obj_size.new_tensor([[0, 0, 1], [0, 0, -1], [0, 1, 0],
                                      [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
        offset = offset.view(1, 6, 3) / 2
        surface_3d = (offset *
                      obj_size.view(batch_size, 1, 3).repeat(1, 6, 1)).reshape(
                          -1, 3)

        # Get the object line center
        offset = obj_size.new_tensor([[1, 0, 1], [-1, 0, 1], [0, 1, 1],
                                      [0, -1, 1], [1, 0, -1], [-1, 0, -1],
                                      [0, 1, -1], [0, -1, -1], [1, 1, 0],
                                      [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
        offset = offset.view(1, 12, 3) / 2

        line_3d = (offset *
                   obj_size.view(batch_size, 1, 3).repeat(1, 12, 1)).reshape(
                       -1, 3)

        surface_rot = rot_mat_T.repeat(6, 1, 1)
        surface_3d = torch.matmul(surface_3d.unsqueeze(-2),
                                  surface_rot).squeeze(-2)
        surface_center = center.repeat(1, 6, 1).reshape(-1, 3) + surface_3d

        line_rot = rot_mat_T.repeat(12, 1, 1)
        line_3d = torch.matmul(line_3d.unsqueeze(-2), line_rot).squeeze(-2)
        line_center = center.repeat(1, 12, 1).reshape(-1, 3) + line_3d

        return surface_center, line_center
