import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    from pytorch3d.ops import box3d_overlap
except ImportError:
    box3d_overlap = None


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """Return the rotation matrices for one of the rotations about an axis of
    which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == 'X':
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == 'Y':
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == 'Z':
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError('letter must be either X, Y or Z.')

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray,
                           convention: str) -> np.ndarray:
    """Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as array of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError('Invalid input euler angles.')
    if len(convention) != 3:
        raise ValueError('Convention must have 3 letters.')
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f'Invalid convention {convention}.')
    for letter in convention:
        if letter not in ('X', 'Y', 'Z'):
            raise ValueError(f'Invalid letter {letter} in convention string.')
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, np.split(euler_angles, 3, axis=-1))
    ]
    matrices = [x.squeeze(axis=-3) for x in matrices]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])


def euler_to_matrix_np(euler):
    """Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler (np.ndarray) : (..., 3)
    Returns:
        np.ndarray : (..., 3, 3)
    """
    # euler: N*3 np array
    euler_tensor = torch.tensor(euler)
    matrix_tensor = euler_angles_to_matrix(euler_tensor, 'ZXY')
    return np.array(matrix_tensor)


def is_inside_box(points, center, size, rotation_mat):
    """Check if points are inside a 3D bounding box.

    Args:
        points: 3D points, numpy array of shape (n, 3).
        center: center of the box, numpy array of shape (3, ).
        size: size of the box, numpy array of shape (3, ).
        rotation_mat: rotation matrix of the box, numpy array of shape (3, 3).
    Returns:
        Boolean array of shape (n, ) indicating if each point is inside the box.
    """
    assert points.shape[1] == 3, 'points should be of shape (n, 3)'
    points = np.array(points)  # n,3
    center = np.array(center)  # n, 3
    size = np.array(size)  # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (
        3,
        3,
    ), f'R should be shape (3,3), but got {rotation_mat.shape}'
    # pcd_local = (rotation_mat.T @ (points - center).T).T  The expressions are equivalent
    pcd_local = (points - center) @ rotation_mat  # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return ((pcd_local[:, 0] <= 1)
            & (pcd_local[:, 1] <= 1)
            & (pcd_local[:, 2] <= 1))


def normalize_box(scene_pcd, embodied_scan_bbox):
    """Find the smallest 6 DoF box that covers these points which 9 DoF box
    covers.

    Args:
        scene_pcd (Tensor / ndarray):
             (..., 3)
        embodied_scan_bbox (Tensor / ndarray):
             (9,) 9 DoF box

    Returns:
        Tensor: Transformed 3D box of shape (N, 8, 3).
    """

    bbox = np.array(embodied_scan_bbox)
    orientation = euler_to_matrix_np(bbox[np.newaxis, 6:])[0]
    position = np.array(bbox[:3])
    size = np.array(bbox[3:6])
    obj_mask = np.array(
        is_inside_box(scene_pcd[:, :3], position, size, orientation),
        dtype=bool,
    )
    obj_pc = scene_pcd[obj_mask]

    # resume the same if there's None
    if obj_pc.shape[0] < 1:
        return embodied_scan_bbox[:6]
    xmin = np.min(obj_pc[:, 0])
    ymin = np.min(obj_pc[:, 1])
    zmin = np.min(obj_pc[:, 2])
    xmax = np.max(obj_pc[:, 0])
    ymax = np.max(obj_pc[:, 1])
    zmax = np.max(obj_pc[:, 2])
    bbox = np.array([
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2,
        xmax - xmin,
        ymax - ymin,
        zmax - zmin,
    ])
    return bbox


def __9dof_to_6dof__(pcd_data, bbox_):
    # that's a kind of loss of information, so we don't recommend
    return normalize_box(pcd_data, bbox_)


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
    eight_corners_x = (torch.tensor([1, 1, 1, 1, -1, -1, -1, -1],
                                    device=device).unsqueeze(0).repeat(
                                        n_box, 1))  # shape (N, 8)
    eight_corners_y = (torch.tensor([1, 1, -1, -1, 1, 1, -1, -1],
                                    device=device).unsqueeze(0).repeat(
                                        n_box, 1))  # shape (N, 8)
    eight_corners_z = (torch.tensor([1, -1, -1, 1, 1, -1, -1, 1],
                                    device=device).unsqueeze(0).repeat(
                                        n_box, 1))  # shape (N, 8)
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


def euler_iou3d_corners(boxes1, boxes2):
    rows = boxes1.shape[0]
    cols = boxes2.shape[0]
    if rows * cols == 0:
        return boxes1.new(rows, cols)

    _, iou3d = box3d_overlap(boxes1, boxes2)
    return iou3d


def euler_iou3d_bbox(center1, size1, rot1, center2, size2, rot2):
    """Calculate the 3D IoU between two grounps of 9DOF bounding boxes.

    Args:
        center1 (Tensor): (n, cx, cy, cz) of grounp1.
        size1 (Tensor): (n, l, w, h) of grounp1.
        rot1 (Tensor): rot matrix of grounp1.
        center1 (Tensor): (m, cx, cy, cz) of grounp2.
        size1 (Tensor): (m, l, w, h) of grounp2.
        rot1 (Tensor): rot matrix of grounp2.

    Returns:
        numpy.ndarray: (n, m)the 3D IoU
    """
    if torch.cuda.is_available():
        center1 = center1.cuda()
        size1 = size1.cuda()
        rot1 = rot1.cuda()
        center2 = center2.cuda()
        size2 = size2.cuda()
        rot2 = rot2.cuda()
    corners1 = bbox_to_corners(center1, size1, rot1)
    corners2 = bbox_to_corners(center2, size2, rot2)
    if torch.cuda.is_available():
        result = euler_iou3d_corners(corners1, corners2)
        try:
            result = result.detach().cpu()
        except:
            print("Failed to transform")
        torch.cuda.empty_cache()
    return result.numpy()
