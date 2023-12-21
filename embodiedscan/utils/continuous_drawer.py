import os

import cv2
import numpy as np
import open3d as o3d

from .utils import _9dof_to_box, draw_camera, from_depth_to_point


class ContinuousDrawer:
    """Visualization tool for Continuous 3D Object Detection task.

    This class serves as the API for visualizing Continuous 3D Object
    Detection task.

    Args:
        dataset (str): Name of composed raw dataset, one of
            scannet/3rscan/matterport3d.
        dir (str): Root path of the dataset.
        scene (dict): Annotation of the selected scene.
        classes (list): Class information.
        color_selector (ColorMap): ColorMap for visualization.
        start_idx (int) : Index of the frame which the task starts.
        pcd_downsample (int) : The rate of downsample.
    """

    def __init__(self, dataset, dir, scene, classes, color_selector, start_idx,
                 pcd_downsample):
        self.dir = dir
        self.dataset = dataset
        self.scene = scene
        self.classes = classes
        self.color_selector = color_selector
        self.idx = start_idx
        self.downsample = pcd_downsample
        self.camera = None
        self.occupied = np.zeros((len(self.scene['instances']), ), dtype=bool)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(262, self.draw_next)  # Right Arrow
        self.vis.register_key_callback(ord('D'), self.draw_next)
        self.vis.register_key_callback(ord('N'), self.draw_next)

    def begin(self):
        """Some preparations before starting the rendering."""
        print('Press N/D/Right Arrow to draw next frame.')
        print('Press Q to close the window and quit.')
        print('Please wait for a few seconds after rendering some frames,',
              'or the program may crash.')
        s = self.scene['sample_idx'].split('/')
        self.occupied = np.zeros((len(self.scene['instances']), ), dtype=bool)
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s
        if dataset == 'scannet':
            pcdpath = os.path.join(self.dir, 'scans', region,
                                   f'{region}_vh_clean.ply')
        elif dataset == '3rscan':
            pcdpath = os.path.join(self.dir, region, 'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            pcdpath = os.path.join(self.dir, building, 'region_segmentations',
                                   f'{region}.ply')
        else:
            raise NotImplementedError
        mesh = o3d.io.read_triangle_mesh(pcdpath, True)
        mesh.transform(self.scene['axis_align_matrix'])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.vis.create_window()
        self.vis.add_geometry(mesh)
        self.vis.add_geometry(frame)
        ctr = self.vis.get_view_control()
        self.view_param = ctr.convert_to_pinhole_camera_parameters()
        self.vis.remove_geometry(mesh)
        self.draw_next(self.vis)

    def draw_next(self, vis):
        """Render the next frame.

        Args:
            vis (open3d.visualization.VisualizerWithKeyCallback): Visualizer.
        """
        if self.idx >= len(self.scene['images']):
            print('No more images')
            return

        img = self.scene['images'][self.idx]
        img_path = img['img_path']
        img_path = os.path.join(self.dir, img_path[img_path.find('/') + 1:])
        depth_path = img['depth_path']
        depth_path = os.path.join(self.dir,
                                  depth_path[depth_path.find('/') + 1:])
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(img_path)
        rgb_img = rgb_img[:, :, ::-1]
        axis_align_matrix = self.scene['axis_align_matrix']
        extrinsic = axis_align_matrix @ img['cam2global']
        if 'cam2img' in img:
            intrinsic = img['cam2img']
        else:
            intrinsic = self.scene['cam2img']
        if 'depth_cam2img' in img:
            depth_intrinsic = img['depth_cam2img']
        else:
            depth_intrinsic = self.scene['depth_cam2img']
        depth_shift = 1000.0
        if self.dataset == 'matterport3d':
            depth_shift = 4000.0
        mask = (depth_img > 0).flatten()
        depth_img = depth_img.astype(np.float32) / depth_shift
        points, colors = from_depth_to_point(rgb_img, depth_img, mask,
                                             intrinsic, depth_intrinsic,
                                             extrinsic)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[::self.downsample])
        pc.colors = o3d.utility.Vector3dVector(colors[::self.downsample])
        vis.add_geometry(pc)
        if self.camera is not None:
            vis.remove_geometry(self.camera)
        self.camera = draw_camera(extrinsic)
        vis.add_geometry(self.camera)

        for ins_idx in img['visible_instance_ids']:
            if self.occupied[ins_idx]:
                continue
            self.occupied[ins_idx] = True
            instance = self.scene['instances'][ins_idx]
            box = _9dof_to_box(instance['bbox_3d'],
                               self.classes[instance['bbox_label_3d'] - 1],
                               self.color_selector)
            vis.add_geometry(box)

        self.idx += 1
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.view_param)
        vis.update_renderer()
        vis.poll_events()
        vis.run()
