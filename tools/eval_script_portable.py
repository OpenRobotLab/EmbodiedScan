# Copyright (c) OpenRobotLab. All rights reserved.
import argparse
from typing import Union

import mmengine
import numpy as np
import torch
from mmengine.logging import print_log
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import euler_angles_to_matrix
from terminaltables import AsciiTable


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


class EulerDepthInstance3DBoxes:
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
        self.with_yaw = with_yaw

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __getitem__(self, item: Union[int, slice, np.ndarray, torch.Tensor]):
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

    @property
    def dims(self) -> torch.Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

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
        assert isinstance(boxes1, EulerDepthInstance3DBoxes)
        assert isinstance(boxes2, EulerDepthInstance3DBoxes)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('results_file', help='the results pkl file')
    parser.add_argument('ann_file', help='annoations json file')

    parser.add_argument('--iou_thr',
                        type=list,
                        default=[0.25, 0.5],
                        help='the IoU threshold during evaluation')

    args = parser.parse_args()
    return args


def ground_eval(gt_annos, det_annos, iou_thr):

    assert len(det_annos) == len(gt_annos)

    pred = {}
    gt = {}

    object_types = [
        'Easy', 'Hard', 'View-Dep', 'View-Indep', 'Unique', 'Multi', 'Overall'
    ]

    for t in iou_thr:
        for object_type in object_types:
            pred.update({object_type + '@' + str(t): 0})
            gt.update({object_type + '@' + str(t): 1e-14})

    for sample_id in range(len(det_annos)):
        det_anno = det_annos[sample_id]
        gt_anno = gt_annos[sample_id]['ann_info']

        bboxes = det_anno['bboxes_3d']
        gt_bboxes = gt_anno['gt_bboxes_3d']
        bboxes = EulerDepthInstance3DBoxes(bboxes, origin=(0.5, 0.5, 0.5))
        gt_bboxes = EulerDepthInstance3DBoxes(gt_bboxes,
                                              origin=(0.5, 0.5, 0.5))
        scores = bboxes.tensor.new_tensor(
            det_anno['scores_3d'])  # (num_query, )

        view_dep = gt_anno['is_view_dep']
        hard = gt_anno['is_hard']
        unique = gt_anno['is_unique']

        box_index = scores.argsort(dim=-1, descending=True)[:10]
        top_bboxes = bboxes[box_index]

        iou = top_bboxes.overlaps(top_bboxes, gt_bboxes)  # (num_query, 1)

        for t in iou_thr:
            threshold = iou > t
            found = int(threshold.any())
            if view_dep:
                gt['View-Dep@' + str(t)] += 1
                pred['View-Dep@' + str(t)] += found
            else:
                gt['View-Indep@' + str(t)] += 1
                pred['View-Indep@' + str(t)] += found
            if hard:
                gt['Hard@' + str(t)] += 1
                pred['Hard@' + str(t)] += found
            else:
                gt['Easy@' + str(t)] += 1
                pred['Easy@' + str(t)] += found
            if unique:
                gt['Unique@' + str(t)] += 1
                pred['Unique@' + str(t)] += found
            else:
                gt['Multi@' + str(t)] += 1
                pred['Multi@' + str(t)] += found

            gt['Overall@' + str(t)] += 1
            pred['Overall@' + str(t)] += found

    header = ['Type']
    header.extend(object_types)
    ret_dict = {}

    for t in iou_thr:
        table_columns = [['results']]
        for object_type in object_types:
            metric = object_type + '@' + str(t)
            value = pred[metric] / max(gt[metric], 1)
            ret_dict[metric] = value
            table_columns.append([f'{value:.4f}'])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table)

    return ret_dict


def main():
    args = parse_args()
    preds = mmengine.load(args.results_file)['results']
    annotations = mmengine.load(args.ann_file)
    assert len(preds) == len(annotations)
    ground_eval(annotations, preds, args.iou_thr)


if __name__ == '__main__':
    main()
