import os
import pickle

import cv2
import numpy as np
import open3d as o3d

from .utils import (_9dof_to_box, _box_add_thickness, draw_camera,
                    from_depth_to_point)


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
        id_to_index (dict): Mapping class id to the index of class names.
        color_selector (ColorMap): ColorMap for visualization.
        start_idx (int) : Index of the frame which the task starts.
        pcd_downsample (int) : The rate of downsample.
    """

    def __init__(self, dataset, dir, scene, classes, id_to_index,
                 color_selector, start_idx, pcd_downsample, thickness):
        self.dir = dir
        self.dataset = dataset
        self.scene = scene
        self.classes = classes
        self.color_selector = color_selector
        self.id_to_index = id_to_index
        self.idx = start_idx
        self.downsample = pcd_downsample
        self.thickness = thickness
        self.camera = None
        self.demo = False
        self.occupied = np.zeros((len(self.scene['instances']), ), dtype=bool)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(262, self.draw_next)  # Right Arrow
        self.vis.register_key_callback(ord('D'), self.draw_next)
        self.vis.register_key_callback(ord('N'), self.draw_next)
        self.vis.register_key_callback(256, self.close)

    def begin(self):
        """Some preparations before starting the rendering."""
        print('Press N/D/Right Arrow to draw next frame.')
        print('Press Q to close the window and quit.')
        print("When you've rendered a lot of frames, the exit can become",
              'very slow because the program needs time to free up space.')
        print('You can also press Esc to close window immediately,',
              'which may result in a segmentation fault.')
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
            self.demo = True
            self.drawed_boxes = []
            pcdpath = None
            camera_config_path = os.path.join(self.dir, region, 'camera.json')
            cam = o3d.io.read_pinhole_camera_parameters(camera_config_path)
        if pcdpath is None:
            self.vis.create_window(width=cam.intrinsic.width,
                                   height=cam.intrinsic.height)
            ctr = self.vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(cam)
            self.view_param = cam
        else:
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
            cam_points = draw_camera(extrinsic, return_points=True)
            self.camera.points = cam_points
            vis.update_geometry(self.camera)
        else:
            self.camera = draw_camera(extrinsic)
            vis.add_geometry(self.camera)

        if self.demo:
            for box in self.drawed_boxes:
                vis.remove_geometry(box)
            self.drawed_boxes = []
        for ins_idx in img['visible_instance_ids']:
            if self.occupied[ins_idx]:
                continue
            self.occupied[ins_idx] = True
            instance = self.scene['instances'][ins_idx]
            box = _9dof_to_box(
                instance['bbox_3d'],
                self.classes[self.id_to_index[instance['bbox_label_3d']]],
                self.color_selector)
            box = _box_add_thickness(box, self.thickness)
            for item in box:
                vis.add_geometry(item)
                if self.demo:
                    self.drawed_boxes.append(item)

        self.idx += 1
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.view_param)
        vis.update_renderer()
        vis.poll_events()
        vis.run()

    def close(self, vis):
        """Close the visualizer.

        Args:
            vis (open3d.visualization.VisualizerWithKeyCallback): Visualizer.
        """
        vis.clear_geometries()
        vis.destroy_window()
        vis.close()


class ContinuousOccupancyDrawer:
    """Visualization tool for Continuous Occupancy Prediction task.

    This class serves as the API for visualizing Continuous 3D Object
    Detection task.

    Args:
        dataset (str): Name of composed raw dataset, one of
            scannet/3rscan/matterport3d.
        dir (str): Root path of the dataset.
        scene (dict): Annotation of the selected scene.
        classes (list): Class information.
        id_to_index (dict): Mapping class id to the index of class names.
        color_selector (ColorMap): ColorMap for visualization.
        start_idx (int) : Index of the frame which the task starts.
    """

    def __init__(self, dataset, dir, scene, classes, id_to_index,
                 color_selector, start_idx):
        self.dir = dir
        self.dataset = dataset
        self.scene = scene
        self.classes = classes
        self.id_to_index = id_to_index
        self.color_selector = color_selector
        self.idx = start_idx
        self.camera = None

        if dataset == 'matterport3d':
            _, building, region = scene['sample_idx'].split('/')
        else:
            _, region = scene['sample_idx'].split('/')

        if dataset == 'scannet':
            self.occ_path = os.path.join(self.dir, 'scans', region,
                                         'occupancy', 'occupancy.npy')
            self.mask_path = os.path.join(self.dir, 'scans', region,
                                          'occupancy', 'visible_occupancy.pkl')
        elif dataset == '3rscan':
            self.occ_path = os.path.join(self.dir, region, 'occupancy',
                                         'occupancy.npy')
            self.mask_path = os.path.join(self.dir, region, 'occupancy',
                                          'visible_occupancy.pkl')
        elif dataset == 'matterport3d':
            self.occ_path = os.path.join(self.dir, building, 'occupancy',
                                         f'occupancy_{region}.npy')
            self.mask_path = os.path.join(self.dir, building, 'occupancy',
                                          f'visible_occupancy_{region}.pkl')
        else:
            raise NotImplementedError

        self.occupied = np.zeros((len(self.scene['instances']), ), dtype=bool)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(262, self.draw_next)  # Right Arrow
        self.vis.register_key_callback(ord('D'), self.draw_next)
        self.vis.register_key_callback(ord('N'), self.draw_next)
        self.vis.register_key_callback(256, self.close)

    def begin(self):
        """Some preparations before starting the rendering."""
        print('Press N/D/Right Arrow to draw next frame.')
        print('Press Q to close the window and quit.')
        print("When you've rendered a lot of frames, the exit can become",
              'very slow because the program needs time to free up space.')
        print('You can also press Esc to close window immediately,',
              'which may result in a segmentation fault.')
        self.gt = np.load(self.occ_path)
        with open(self.mask_path, 'rb') as f:
            self.mask = pickle.load(f)

        point_cloud_range = [-3.2, -3.2, -1.28 + 0.5, 3.2, 3.2, 1.28 + 0.5]
        occ_size = [40, 40, 16]
        self.grid_size = 0.16

        self.points = np.zeros((self.gt.shape[0], 6), dtype=float)
        self.gird_id = np.ones(occ_size, dtype=int) * -1
        self.visible_mask = np.zeros((self.gt.shape[0], ), dtype=bool)
        for i in range(self.gt.shape[0]):
            x, y, z, label_id = self.gt[i]
            self.gird_id[x, y, z] = i
            label_id = int(label_id)
            if label_id == 0:
                label = 'object'
            else:
                label = self.classes[self.id_to_index[label_id]]
            color = self.color_selector.get_color(label)
            color = [x / 255.0 for x in color]
            self.points[i][:3] = [
                x * self.grid_size + point_cloud_range[0] + self.grid_size / 2,
                y * self.grid_size + point_cloud_range[1] + self.grid_size / 2,
                z * self.grid_size + point_cloud_range[2] + self.grid_size / 2
            ]
            self.points[i][3:] = color

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.points[:, 3:])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=self.grid_size)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.vis.create_window()
        self.vis.add_geometry(voxel_grid)
        self.vis.add_geometry(frame)
        ctr = self.vis.get_view_control()
        self.view_param = ctr.convert_to_pinhole_camera_parameters()
        self.voxel_grid = voxel_grid
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
        extrinsic = self.scene['axis_align_matrix'] @ img['cam2global']

        mask = self.mask[self.idx]['visible_occupancy']
        visible_ids = np.unique(self.gird_id[mask])
        visible_ids = visible_ids[visible_ids >= 0]
        self.visible_mask[visible_ids] = True
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self.points[self.visible_mask][:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            self.points[self.visible_mask][:, 3:])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=self.grid_size)

        if self.camera is not None:
            cam_points = draw_camera(extrinsic, return_points=True)
            self.camera.points = cam_points
            vis.update_geometry(self.camera)
        else:
            self.camera = draw_camera(extrinsic)
            vis.add_geometry(self.camera)

        self.voxel_grid.clear()
        vis.update_geometry(self.voxel_grid)
        vis.remove_geometry(self.voxel_grid)
        vis.add_geometry(voxel_grid)
        self.voxel_grid = voxel_grid
        self.idx += 1
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.view_param)
        vis.update_renderer()
        vis.poll_events()
        vis.run()

    def close(self, vis):
        """Close the visualizer.

        Args:
            vis (open3d.visualization.VisualizerWithKeyCallback): Visualizer.
        """
        vis.clear_geometries()
        vis.destroy_window()
        vis.close()
