import torch
from typing import Union, Tuple
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
try:
    from pytorch3d.ops import box3d_overlap
except ImportError:
    print("warning: failed to import pytorch3d")
    box3d_overlap = None

if box3d_overlap is not None:
    from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
else:
    def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
        """Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
        """Convert rotations given as Euler angles in radians to rotation
        matrices.

        Args:
            euler_angles: Euler angles in radians as tensor of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            _axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]
        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def euler_to_matrix_np(euler):
    # euler: N*3 np array
    euler_tensor = torch.tensor(euler)
    matrix_tensor = euler_angles_to_matrix(euler_tensor, 'ZXY')
    return matrix_tensor.numpy()

def bbox_to_corners(centers, sizes, rot_mat: torch.Tensor) -> torch.Tensor:
    """Transform bbox parameters to the 8 corners.

    Args:
        bbox (Tensor): 3D box of shape (N, 6) or (N, 7) or (N, 9).

    Returns:
        Tensor: Transformed 3D box of shape (N, 8, 3).
    """
    device = centers.device
    use_batch = False
    if len(centers.shape) == 3:
        use_batch = True
        batch_size, n_proposals = centers.shape[0], centers.shape[1]
        centers = centers.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        rot_mat = rot_mat.reshape(-1, 3, 3)
    
    n_box = centers.shape[0]
    if use_batch:
        assert n_box == batch_size * n_proposals
    centers = centers.unsqueeze(1).repeat(1, 8, 1)  # shape (N, 8, 3)
    half_sizes = sizes.unsqueeze(1).repeat(1, 8, 1) / 2  # shape (N, 8, 3)
    eight_corners_x = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1],
                                   device=device).unsqueeze(0).repeat(n_box, 1)  # shape (N, 8)
    eight_corners_y = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1],
                                   device=device).unsqueeze(0).repeat(
                                       n_box, 1)  # shape (N, 8)
    eight_corners_z = torch.tensor([1, -1, -1, 1, 1, -1, -1, 1],
                                   device=device).unsqueeze(0).repeat(
                                       n_box, 1)  # shape (N, 8)
    eight_corners = torch.stack(
        (eight_corners_x, eight_corners_y, eight_corners_z),
        dim=-1)  # shape (N, 8, 3)
    eight_corners = eight_corners * half_sizes  # shape (N, 8, 3)
    # rot_mat: (N, 3, 3), eight_corners: (N, 8, 3)
    rotated_corners = torch.matmul(eight_corners,
                                   rot_mat.transpose(1, 2))  # shape (N, 8, 3)
    res = centers + rotated_corners
    if use_batch:
        res = res.reshape(batch_size, n_proposals, 8, 3)
    return res

def chamfer_distance(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_weight: Union[torch.Tensor, float] = 1.0,
        dst_weight: Union[torch.Tensor, float] = 1.0,
        criterion_mode: str = 'l2',
        reduction: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Chamfer Distance of two sets.

    Args:
        src (Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (Tensor or float): Weight of source loss. Defaults to 1.0.
        dst_weight (Tensor or float): Weight of destination loss.
            Defaults to 1.0.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are 'smooth_l1', 'l1' or 'l2'. Defaults to 'l2'.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
            Defaults to 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (Tensor): The min distance
              from source to destination.
            - loss_dst (Tensor): The min distance
              from destination to source.
            - indices1 (Tensor): Index the min distance point
              for each point in source to destination.
            - indices2 (Tensor): Index the min distance point
              for each point in destination to source.
    """
    if len(src.shape) == 4:
        src = src.reshape(-1, 8, 3)
    if len(dst.shape) == 4:
        dst = dst.reshape(-1, 8, 3)

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2


def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
    ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
    of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
        bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "giou" (generalized
            intersection over union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Defaults to False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Defaults to 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 10, 10],
        >>>     [10, 10, 10, 20, 20, 20],
        >>>     [32, 32, 32, 38, 40, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 20, 20],
        >>>     [0, 10, 10, 10, 19, 20],
        >>>     [10, 10, 10, 20, 20, 20],
        >>> ])
        >>> overlaps = axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 6)
        >>> nonempty = torch.FloatTensor([[0, 0, 0, 10, 9, 10]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def euler_iou3d(boxes1, boxes2):
    rows = boxes1.shape[0]
    cols = boxes2.shape[0]
    if rows * cols == 0:
        return boxes1.new(rows, cols)

    _, iou3d = box3d_overlap(boxes1, boxes2)
    return iou3d

def euler_iou3d_split(center1, size1, rot1, center2, size2, rot2):
    device = center1.device
    center1 = center1.cuda()
    size1 = size1.cuda()
    rot1 = rot1.cuda()
    center2 = center2.cuda()
    size2 = size2.cuda()
    rot2 = rot2.cuda()
    corners1 = bbox_to_corners(center1, size1, rot1)
    corners2 = bbox_to_corners(center2, size2, rot2)
    return euler_iou3d(corners1, corners2).to(device)