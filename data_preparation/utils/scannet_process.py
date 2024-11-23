import os

import numpy as np
from plyfile import PlyData


def process_scannet(scan_id, data_root, scannet_matrix):
    scan_ply_path = os.path.join(f'{data_root}/scans', scan_id,
                                 scan_id + '_vh_clean_2.labels.ply')
    data_color = PlyData.read(
        os.path.join(f'{data_root}/scans', scan_id,
                     scan_id + '_vh_clean_2.ply'))
    data = PlyData.read(scan_ply_path)
    x = np.asarray(data.elements[0].data['x']).astype(np.float32)
    y = np.asarray(data.elements[0].data['y']).astype(np.float32)
    z = np.asarray(data.elements[0].data['z']).astype(np.float32)
    pc = np.stack([x, y, z], axis=1)
    r = np.asarray(data_color.elements[0].data['red'])
    g = np.asarray(data_color.elements[0].data['green'])
    b = np.asarray(data_color.elements[0].data['blue'])
    pc_color = (np.stack([r, g, b], axis=1) / 255.0).astype(np.float32)
    axis_align_matrix = scannet_matrix[scan_id]
    pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
    pts[:, :3] = pc
    pc = np.dot(pts, axis_align_matrix.transpose())[:, :3]
    return pc, pc_color, pc_color[:, 0]
