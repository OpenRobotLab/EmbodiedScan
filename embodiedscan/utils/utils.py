import cv2
import numpy as np
import open3d as o3d


def from_depth_to_point(rgb, depth, mask, intrinsic, depth_intrinsic,
                        extrinsic):
    h2, w2, _ = rgb.shape
    h, w = depth.shape
    depth_intrinsic = np.linalg.inv(depth_intrinsic)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    z = depth.reshape((1, -1))[:, :]
    rgb_resized = cv2.resize(rgb, (w, h))
    color = rgb_resized.reshape(-1, 3) / 255.0
    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = depth_intrinsic[:3, :3] @ p_2d
    pc = pc * z
    pc = np.concatenate([pc, np.ones_like(x)], axis=0)
    pc = extrinsic @ pc
    z_mask = pc[2, :] < 1.8
    mask = np.logical_and(mask, z_mask)
    pc = pc.transpose()
    pc = pc[mask]
    color = color[mask]
    return pc[:, :3], color


def _9dof_to_box(box, label, color_selector):
    if isinstance(box, list):
        box = np.array(box)
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(
        rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    color = color_selector.get_color(label)
    color = [x / 255.0 for x in color]
    geo.color = color
    return geo


def draw_camera(camera_pose, camera_size=0.5, return_points=False):
    # camera_pose : 4*4 camera to world
    point = np.array([[0, 0, 0], [-camera_size, -camera_size, camera_size * 2],
                      [camera_size, -camera_size, camera_size * 2],
                      [-camera_size, camera_size, camera_size * 2],
                      [camera_size, camera_size, camera_size * 2]])
    pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point))
    pc.transform(camera_pose)
    if return_points:
        return pc.points
    color = (100 / 255.0, 149 / 255.0, 237 / 255.0)
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0,
                                                                   3], [0, 4],
                                                  [1, 2], [1, 3], [2, 4],
                                                  [3, 4]])
    lines_pcd.points = pc.points
    lines_pcd.paint_uniform_color(color)
    return lines_pcd
