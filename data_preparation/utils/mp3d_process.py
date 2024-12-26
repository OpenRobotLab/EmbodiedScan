import numpy as np
from plyfile import PlyData


def process_mp3d(new_scan_id: str, data_root: str,
                 axis_align_matrix_dict: dict, mapping: dict):
    """Process matterport3d data.

    Args:
        new_scan_id (str): processed ID of the matterport3d scan.
        data_root (str): Root directory of the matterport3d dataset.
        axis_align_matrix_dict (dict): Dict of axis alignment matrices
            for each scan.
        mapping (dict) : Dict of mapping names.

    Returns:
        tuple : point_xyz and point_rgb infos.
    """
    axis_align_matrix = axis_align_matrix_dict[new_scan_id]

    scan_id, region_id = (
        new_scan_id.split('_region')[0],
        'region' + new_scan_id.split('_region')[1],
    )
    a = PlyData.read(
        f'{data_root}/{mapping[scan_id]}/region_segmentations/{region_id}.ply')
    v = np.array([list(x) for x in a.elements[0]])

    pc = np.ascontiguousarray(v[:, :3])
    pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
    pts[:, :3] = pc
    pc = np.dot(pts, axis_align_matrix.transpose())[:, :3].astype(np.float32)
    colors = np.ascontiguousarray(v[:, -3:])
    colors = colors / 255.0
    colors = colors.astype(np.float32)
    return pc, colors, colors[:, 0]
