import numpy as np
from scipy.linalg import eigh
from scipy.spatial import ConvexHull


def interpolate_bbox_points(bbox, granularity=0.2, return_size=False):
    """Get the surface points of a 3D bounding box.

    Args:
        bbox: an open3d.geometry.OrientedBoundingBox object.
        granularity: the roughly desired distance between two adjacent surface points.
        return_size: if True, return m1, m2, m3 as well.
    Returns:
        M x 3 numpy array of Surface points of the bounding box
        (m1, m2, m3): if return_size is True, return the number for each dimension.)
    """
    corners = np.array(bbox.get_box_points())
    v1, v2, v3 = (
        corners[1] - corners[0],
        corners[2] - corners[0],
        corners[3] - corners[0],
    )
    l1, l2, l3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    assert (np.allclose(v1.dot(v2), 0) and np.allclose(v2.dot(v3), 0)
            and np.allclose(v3.dot(v1), 0))
    transformation_matrix = np.column_stack((v1, v2, v3))
    m1, m2, m3 = l1 / granularity, l2 / granularity, l3 / granularity
    m1, m2, m3 = int(np.ceil(m1)), int(np.ceil(m2)), int(np.ceil(m3))
    coords = np.array(
        np.meshgrid(np.arange(m1 + 1), np.arange(m2 + 1),
                    np.arange(m3 + 1))).T.reshape(-1, 3)
    condition = ((coords[:, 0] == 0)
                 | (coords[:, 0] == m1 - 1)
                 | (coords[:, 1] == 0)
                 | (coords[:, 1] == m2 - 1)
                 | (coords[:, 2] == 0)
                 | (coords[:, 2] == m3 - 1))
    surface_points = coords[condition].astype(
        'float32')  # keep only the points on the surface
    surface_points /= np.array([m1, m2, m3])
    mapped_coords = surface_points @ transformation_matrix
    mapped_coords = mapped_coords.reshape(-1, 3) + corners[0]
    if return_size:
        return mapped_coords, (m1, m2, m3)
    return mapped_coords


def check_bboxes_visibility(bboxes,
                            depth_map,
                            depth_intrinsic,
                            extrinsic,
                            corners_only=True,
                            granularity=0.2):
    """Check the visibility of 3D bounding boxes in a depth map.

    Args:
        bboxes: a list of N open3d.geometry.OrientedBoundingBox
        depth_map: depth map, numpy array of shape (h, w).
        depth_intrinsic: numpy array of shape (4, 4).
        extrinsic: w2c. numpy array of shape (4, 4).
        corners_only: if True, only check the corners of the bounding boxes.
        granularity: the roughly desired distance between two adjacent surface points.
    Returns:
        Boolean array of shape (N, ) indicating the visibility of each bounding box.
    """
    if corners_only:
        points = [box.get_box_points() for box in bboxes]
        num_points_per_bbox = [8] * len(bboxes)
        points = np.concatenate(points, axis=0)  # shape (N*8, 3)
    else:
        points, num_points_per_bbox, num_points_to_view = [], [], []
        for bbox in bboxes:
            interpolated_points, (m1, m2, m3) = interpolate_bbox_points(
                bbox, granularity=granularity, return_size=True)
            num_points_per_bbox.append(interpolated_points.shape[0])
            points.append(interpolated_points)
            num_points_to_view.append(max(m1 * m2, m1 * m3, m2 * m3))
        points = np.concatenate(points, axis=0)  # shape (\sum Mi, 3)
        num_points_to_view = np.array(num_points_to_view)
    num_points_per_bbox = np.array(num_points_per_bbox)
    visibles = check_point_visibility(points, depth_map, depth_intrinsic,
                                      extrinsic)
    num_visibles = []
    left = 0
    for i, num_points in enumerate(num_points_per_bbox):
        slice_i = visibles[left:left + num_points]
        num_visibles.append(np.sum(slice_i))
        left += num_points
    num_visibles = np.array(num_visibles)
    visibles = num_visibles / num_points_to_view >= 1  # threshold for visibility
    return visibles


def check_point_visibility(points, depth_map, depth_intrinsic, extrinsic):
    """Check the visibility of 3D points in a depth map.

    Args:
        points: 3D points, numpy array of shape (n, 3).
        depth_map: depth map, numpy array of shape (h, w).
        depth_intrinsic: numpy array of shape (4, 4).
        extrinsic: w2c. numpy array of shape (4, 4).
    Returns:
        Boolean array of shape (n, ) indicating the visibility of each point.
    """
    # Project 3D points to 2D image plane
    visibles = np.ones(points.shape[0], dtype=bool)
    points = np.concatenate([points, np.ones_like(points[..., :1])],
                            axis=-1)  # shape (n, 4)
    points = depth_intrinsic @ extrinsic @ points.T  # (4, n)
    xs, ys, zs = points[:3, :]
    visibles &= zs > 0  # remove points behind the camera
    xs, ys = xs / zs, ys / zs  # normalize to image plane
    height, width = depth_map.shape
    visibles &= ((0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
                 )  # remove points outside the image
    xs[(xs < 0) | (xs >= width)] = 0  # avoid index out of range in depth_map
    ys[(ys < 0) | (ys >= height)] = 0
    visibles &= (depth_map[ys.astype(int), xs.astype(int)] > zs
                 )  # remove points occluded by other objects
    return visibles


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
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])


box_corner_vertices = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]


def cal_corners_single(center, size, rotmat):
    center = np.array(center).reshape(3)
    size = np.array(size).reshape(3)
    rotmat = np.array(rotmat).reshape(3, 3)

    relative_corners = np.array(box_corner_vertices)
    relative_corners = 2 * relative_corners - 1
    corners = relative_corners * size / 2.0
    corners = np.dot(corners, rotmat.T).reshape(-1, 3)
    corners += center
    return corners


def cal_corners(center, size, rotmat):
    center = np.array(center).reshape(-1, 3)
    size = np.array(size).reshape(-1, 3)
    rotmat = np.array(rotmat).reshape(-1, 3, 3)
    bsz = center.shape[0]

    relative_corners = np.array(box_corner_vertices)
    relative_corners = 2 * relative_corners - 1
    relative_corners = np.expand_dims(relative_corners, 1).repeat(bsz, axis=1)
    corners = relative_corners * size / 2.0
    corners = corners.transpose(1, 0, 2)
    corners = np.matmul(corners, rotmat.transpose(0, 2, 1)).reshape(-1, 8, 3)
    corners += np.expand_dims(center, 1).repeat(8, axis=1)

    if corners.shape[0] == 1:
        corners = corners[0]
    return corners


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
    center = np.array(center)  # n, 3
    size = np.array(size)  # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (
        3, 3), f'R should be shape (3,3), but got {rotation_mat.shape}'
    # pcd_local = (rotation_mat.T @ (points - center).T).T  The expressions are equivalent
    pcd_local = (points - center) @ rotation_mat  # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return (pcd_local[:, 0] <= 1) & (pcd_local[:, 1] <= 1) & (pcd_local[:, 2]
                                                              <= 1)


def is_inside_box_open3d(points, center, size, rotation_mat):
    import open3d as o3d
    """
        We have verified that the two functions are equivalent.
        but the first one is faster.
    """
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = center
    obb.extent = size
    obb.R = rotation_mat
    points_o3d = o3d.utility.Vector3dVector(points)
    point_indices_within_box = obb.get_point_indices_within_bounding_box(
        points_o3d)
    ret = np.zeros(points.shape[0], dtype=bool)
    ret[point_indices_within_box] = True
    return ret


def compute_bbox_from_points_open3d(points):
    import open3d as o3d

    # 2e-4 seconds for 100 points
    # 1e-3 seconds for 1000 points
    points = np.array(points).astype(np.float32)
    assert points.shape[
        1] == 3, f'points should be of shape (n, 3), but got {points.shape}'
    o3dpoints = o3d.utility.Vector3dVector(points)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3dpoints)

    center = obb.center
    size = obb.extent
    rotation = obb.R

    points_to_check = center + (points - center) * (
        1.0 - 1e-6)  # add a small epsilon to avoid numerical issues
    mask = is_inside_box(points_to_check, center, size, rotation)
    # mask2 = is_inside_box_open3d(points, center, size, rotation)
    # assert (mask == mask2).all(), "mask is different from mask2"
    # assert mask2.all(), (center, size, rotation)
    assert mask.all(), (center, size, rotation)

    return center, size, rotation


def compute_bbox_from_points(points):
    # 7.5e-4 seconds for 100 points
    # 1e-3 seconds for 1000 points
    hull = ConvexHull(points)
    points_on_hull = points[hull.vertices]
    center = points_on_hull.mean(axis=0)

    points_centered = points_on_hull - center
    cov_matrix = np.cov(points_centered, rowvar=False)
    eigvals, eigvecs = eigh(cov_matrix)
    rotation = eigvecs
    # donot use eigvals to compute size, but use the min max projections
    proj = points_centered @ eigvecs
    min_proj = np.min(proj, axis=0)
    max_proj = np.max(proj, axis=0)
    size = max_proj - min_proj
    shift = (max_proj + min_proj) / 2.0
    center = center + shift @ rotation.T

    return center, size, rotation


def check_pcd_similarity(pcd1, pcd2):
    """check whether two point clouds are close enough.

    There might be a permulatation issue, so we use a threshold to check.
    Args:
        pcd1: np.array of shape (n, 3)
        pcd2: np.array of shape (n, 3)
    """
    assert pcd1.shape == pcd2.shape, f'pcd1 and pcd2 should have the same shape, but got {pcd1.shape} and {pcd2.shape}'
    pcd1 = pcd1.astype(np.float64)
    pcd2 = pcd2.astype(np.float64)
    distances_mat = np.sqrt(
        np.sum((pcd1.reshape(1, -1, 3) - pcd2.reshape(-1, 1, 3))**2, axis=-1))
    distances_rowwise = np.min(distances_mat, axis=1)
    distances_colwise = np.min(distances_mat, axis=0)
    threshold = 1e-2
    return (distances_rowwise < threshold).all() and (distances_colwise <
                                                      threshold).all()


if __name__ == '__main__':
    from tqdm import tqdm
    centers = np.random.rand(100, 3) * 10 - 5
    sizes = np.random.rand(100, 3) * 1 + 1
    angle_range = np.pi
    roll, pitch, yaw = np.random.rand(3) * angle_range * 2 - angle_range
    rotation = euler_angles_to_matrix(np.array([yaw, pitch, roll]), 'ZYX')
    corners = cal_corners(centers, sizes, rotation)
    for i in range(100):
        center = centers[i]
        size = sizes[i]
        corners_i = corners[i]
        assert check_pcd_similarity(corners_i,
                                    cal_corners_single(center, size, rotation))
    exit()

    for i in tqdm(range(10000)):
        center = np.random.rand(3) * 10 - 5
        size = np.random.rand(3) * 1 + 1
        angle_range = np.pi
        roll, pitch, yaw = np.random.rand(3) * angle_range * 2 - angle_range
        # center = np.array([0, 0, 0])
        # size = np.array([1, 1, 1])
        # roll, pitch, yaw = 0, 0, 1
        rotation = euler_angles_to_matrix(np.array([yaw, pitch, roll]), 'ZYX')
        corners = cal_corners_single(center, size, rotation)
        # print(points)
        center2, size2, rotation2 = compute_bbox_from_points_open3d(corners)
        corners2 = cal_corners_single(center2, size2, rotation2)
        compare_dict = {
            'center': (center, center2),
            'size': (size, size2),
            'rotation': (rotation, rotation2),
            'corners': (corners, corners2)
        }
        if not check_pcd_similarity(corners, corners2):
            print(compare_dict)
            exit()
        center3, size3, rotation3 = compute_bbox_from_points(corners)
        corners3 = cal_corners_single(center3, size3, rotation3)
        compare_dict = {
            'center': (center, center3),
            'size': (size, size3),
            'rotation': (rotation, rotation3),
            'corners': (corners, corners3)
        }
        if not check_pcd_similarity(corners, corners3):
            for k, v in compare_dict.items():
                print(k, v)
            exit()
    for i in tqdm(range(1000)):
        points = np.random.rand(100, 3) * 10 - 5
        center, size, rotation = compute_bbox_from_points_open3d(points)
        corners = cal_corners_single(center, size, rotation)
        center2, size2, rotation2 = compute_bbox_from_points(points)
        corners2 = cal_corners_single(center2, size2, rotation2)
        vol1 = size[0] * size[1] * size[2]
        vol2 = size2[0] * size2[1] * size2[2]
        if not check_pcd_similarity(corners, corners2):
            print(vol1, vol2)
            print(f'vol1: {vol1}, vol2: {vol2}')
            print(f'center: {center}, center2: {center2}')
            print(f'size: {size}, size2: {size2}')
            print(f'rotation: {rotation}, rotation2: {rotation2}')
            print(f'corners: {corners}, corners2: {corners2}')
            exit()
