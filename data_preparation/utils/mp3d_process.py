import numpy as np
from plyfile import PlyData


def process_mp3d(new_scan_id, data_root, axis_align_matrix_dict, mapping):
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
