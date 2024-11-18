import cv2
import numpy as np
import open3d as o3d
from torch import Tensor

from .line_mesh import LineMesh


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


def _box_add_thickness(box, thickness):
    bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    bbox_lines_width = LineMesh(points=bbox_lines.points,
                                lines=bbox_lines.lines,
                                colors=box.color,
                                radius=thickness)
    results = bbox_lines_width.cylinder_segments
    return results


def _9dof_to_box(box, label=None, color_selector=None, color=None):
    """Convert 9-DoF box from array/tensor to open3d.OrientedBoundingBox.

    Args:
        box (numpy.ndarray|torch.Tensor|List[float]):
            9-DoF box with shape (9,).
        label (int, optional): Label of the box. Defaults to None.
        color_selector (:obj:`ColorSelector`, optional):
            Color selector for boxes. Defaults to None.
        color (tuple[int], optional): Color of the box.
            You can directly specify the color.
            If you do, the color_selector and label will be ignored.
            Defaults to None.
    """
    if isinstance(box, list):
        box = np.array(box)
    if isinstance(box, Tensor):
        box = box.cpu().numpy()
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(
        rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    if color is not None:
        geo.color = [x / 255.0 for x in color]
        return geo

    if label is not None and color_selector is not None:
        color = color_selector.get_color(label)
        color = [x / 255.0 for x in color]
        geo.color = color
    return geo


def nms_filter(pred_results, iou_thr=0.15, score_thr=0.075, topk_per_class=10):
    """Non-Maximum Suppression for 3D Euler boxes. Additionally, only the top-k
    boxes will be kept for each category to avoid redundant boxes in the
    visualization.

    Args:
        pred_results (:obj:`InstanceData`):
            Results predicted by the model.
        iou_thr (float): IoU thresholds for NMS. Defaults to 0.15.
        score_thr (float): Score thresholds.
            Instances with scores below thresholds will not be kept.
            Defaults to 0.075.
        topk_per_class (int): Number of instances kept per category.
            Defaults to 10.

    Returns:
        numpy.ndarray[float], np.ndarray[int]:
            Filtered boxes with shape (N, 9) and labels with shape (N,).
    """
    boxes = pred_results.bboxes_3d
    boxes_tensor = boxes.tensor.cpu().numpy()
    iou = boxes.overlaps(boxes, boxes, eps=1e-5)
    score = pred_results.scores_3d.cpu().numpy()
    label = pred_results.labels_3d.cpu().numpy()
    selected_per_class = dict()

    n = boxes_tensor.shape[0]
    idx = list(range(n))
    idx.sort(key=lambda x: score[x], reverse=True)
    selected_idx = []
    for i in idx:
        if selected_per_class.get(label[i], 0) >= topk_per_class:
            continue
        if score[i] < score_thr:
            continue
        bo = False
        for j in selected_idx:
            if iou[i][j] > iou_thr:
                bo = True
                break
        if not bo:
            selected_idx.append(i)
            if label[i] not in selected_per_class:
                selected_per_class[label[i]] = 1
            else:
                selected_per_class[label[i]] += 1

    return boxes_tensor[selected_idx], label[selected_idx]


def draw_camera(camera_pose, camera_size=0.5, return_points=False):
    """Draw the camera pose in the form of a cone.

    Args:
        camera_pose (numpy.ndarray): 4x4 camera pose from camera to world.
        camera_size (float): Size of the camera cone. Defaults to 0.5.
        return_points (bool): Whether to return the points of the camera cone.
            Defaults to False.

    Returns:
        numpy.ndarray | :obj:`LineSet`:
            if return_points is True, return the points of the camera cone.
            Otherwise, return the camera cone as an open3d.LineSet.
    """
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
