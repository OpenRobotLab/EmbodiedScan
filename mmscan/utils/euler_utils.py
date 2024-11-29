# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    from pytorch3d.ops import box3d_overlap
    from pytorch3d.transforms import (euler_angles_to_matrix,
                                      matrix_to_euler_angles)
except ImportError:
    box3d_overlap = None
    euler_angles_to_matrix = None
    matrix_to_euler_angles = None
from torch import Tensor


class BaseInstance3DBoxes:
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in the
        box is (0.5, 0.5, 0).

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0)
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, \
            ('The box dimension must be 2 and the length of the last '
             f'dimension must be {box_dim}, but got boxes with shape '
             f'{tensor.shape}.')

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding 0 as
            # a fake yaw and set with_yaw to False
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of boxes."""
        return self.tensor.shape

    @property
    def volume(self) -> Tensor:
        """Tensor: A vector with volume of each box in shape (N, )."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self) -> Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self) -> Tensor:
        """Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self) -> Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self) -> Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self) -> Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self) -> Tensor:
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is usually taken
            as the default center.

            The relative position of the centers in different kinds of boxes
            are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar. It is
            recommended to use ``bottom_center`` or ``gravity_center`` for
            clearer usage.

        Returns:
            Tensor: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center

    @property
    def bottom_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self) -> Tensor:
        """Tensor: A tensor with 8 corners of each box in shape (N, 8, 3)."""
        pass

    @property
    def bev(self) -> Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    def in_range_bev(
            self, box_range: Union[Tensor, np.ndarray,
                                   Sequence[float]]) -> Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box in order of (x_min, y_min, x_max, y_max).

        Note:
            The original implementation of SECOND checks whether boxes in a
            range by checking whether the points are in a convex polygon, we
            reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each box is inside the
            reference range.
        """
        in_range_flags = ((self.bev[:, 0] > box_range[0])
                          & (self.bev[:, 1] > box_range[1])
                          & (self.bev[:, 0] < box_range[2])
                          & (self.bev[:, 1] < box_range[3]))
        return in_range_flags

    @abstractmethod
    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray],
               Tuple[Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:``, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        pass

    @abstractmethod
    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, ]] = None
    ) -> Union[Tensor, np.ndarray, None]:
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:``, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        pass

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size
                1x3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
            self, box_range: Union[Tensor, np.ndarray,
                                   Sequence[float]]) -> Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 2] > box_range[2])
                          & (self.tensor[:, 0] < box_range[3])
                          & (self.tensor[:, 1] < box_range[4])
                          & (self.tensor[:, 2] < box_range[5]))
        return in_range_flags

    @abstractmethod
    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: float) -> None:
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor  # velocity

    def nonempty(self, threshold: float = 0.0) -> Tensor:
        """Find boxes that are non-empty.

        A box is considered empty if either of its side is no larger than
        threshold.

        Args:
            threshold (float): The threshold of minimal sizes. Defaults to 0.0.

        Returns:
            Tensor: A binary vector which represents whether each box is empty
            (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def __getitem__(
            self, item: Union[int, slice, np.ndarray,
                              Tensor]) -> 'BaseInstance3DBoxes':
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1),
                                 box_dim=self.box_dim,
                                 with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list: Sequence['BaseInstance3DBoxes']
            ) -> 'BaseInstance3DBoxes':
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (Sequence[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0),
                        box_dim=boxes_list[0].box_dim,
                        with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the specific
            device.
        """
        original_type = type(self)
        return original_type(self.tensor.to(device, *args, **kwargs),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    def cpu(self) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cpu device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cpu device.
        """
        original_type = type(self)
        return original_type(self.tensor.cpu(),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    def cuda(self, *args, **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cuda device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cuda device.
        """
        original_type = type(self)
        return original_type(self.tensor.cuda(*args, **kwargs),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    def clone(self) -> 'BaseInstance3DBoxes':
        """Clone the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    def detach(self) -> 'BaseInstance3DBoxes':
        """Detach the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(self.tensor.detach(),
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a box as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A box of shape (box_dim, ).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1: 'BaseInstance3DBoxes',
                        boxes2: 'BaseInstance3DBoxes') -> Tensor:
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.

        Returns:
            Tensor: Calculated height overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), \
            '"boxes1" and "boxes2" should be in the same type, ' \
            f'but got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    def new_box(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> 'BaseInstance3DBoxes':
        """Create a new box object with data.

        The new box and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, the
            object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor,
                             box_dim=self.box_dim,
                             with_yaw=self.with_yaw)


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
            points (torch.Tensor | np.ndarray | :obj:``, optional):
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
            elif isinstance(points, ):
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


def rotation_3d_in_euler(points, angles, return_mat=False, clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple):
            Vector of angles in shape (N, 3)
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if len(angles.shape) == 1:
        angles = angles.expand(points.shape[:1] + (3, ))
        # angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 2 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_mat_T = euler_angles_to_matrix(angles, 'ZXY')  # N, 3,3
    rot_mat_T = rot_mat_T.transpose(-2, -1)

    if clockwise:
        raise NotImplementedError('clockwise')

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.bmm(points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new
