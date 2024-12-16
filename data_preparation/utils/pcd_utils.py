import os

import numpy as np
import torch
from plyfile import PlyData


def read_mesh_vertices_rgb(filename: str) -> np.ndarray:
    """Read XYZ and RGB for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        np.ndarray: Note that RGB values are in 0-255.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices


def is_inside_box(points: np.ndarray, center: np.ndarray, size: np.ndarray,
                  rotation_mat: np.ndarray) -> np.ndarray:
    """Check if points are inside a 3D bounding box.

    Args:
        points(np.ndarray): 3D points, numpy array of shape (n, 3).
        center(np.ndarray): center of the box, numpy array of shape (3, ).
        size(np.ndarray): size of the box, numpy array of shape (3, ).
        rotation_mat(np.ndarray): rotation matrix of the box,
            numpy array of shape (3, 3).

    Returns:
        np.ndarray: Boolean array of shape (n, ) indicating if each point
            is inside the box.
    """
    assert points.shape[1] == 3, 'points should be of shape (n, 3)'
    center = np.array(center)  # n, 3
    size = np.array(size)  # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (
        3,
        3,
    ), f'R should be shape (3,3), but got {rotation_mat.shape}'
    pcd_local = (points - center) @ rotation_mat  # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return ((pcd_local[:, 0] <= 1)
            & (pcd_local[:, 1] <= 1)
            & (pcd_local[:, 2] <= 1))


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
