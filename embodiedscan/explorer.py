import os
import pickle
from typing import List, Union

import numpy as np
import open3d as o3d

from embodiedscan.visualization.color_selector import ColorMap
from embodiedscan.visualization.continuous_drawer import (
    ContinuousDrawer, ContinuousOccupancyDrawer)
from embodiedscan.visualization.img_drawer import ImageDrawer
from embodiedscan.visualization.utils import _9dof_to_box, _box_add_thickness

DATASETS = ['scannet', '3rscan', 'matterport3d']


class EmbodiedScanExplorer:
    """EmbodiedScan Explorer.

    This class serves as the API for analyze and visualize EmbodiedScan
    dataset with demo data.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        verbose (bool): Whether to print related messages. Defaults to False.
        color_setting (str, optional): Color settings for visualization.
            Defaults to None.
            Accept the path to the setting file like
                embodiedscan/visualization/full_color_map.txt
        thickness (float): Thickness of of the displayed box lines.
    """

    def __init__(self,
                 data_root: Union[dict, List],
                 ann_file: Union[dict, List, str],
                 verbose: bool = False,
                 color_setting: str = None,
                 thickness: float = 0.01):

        if isinstance(ann_file, dict):
            ann_file = list(ann_file.values())
        elif isinstance(ann_file, str):
            ann_file = [ann_file]
        self.ann_files = ann_file

        if isinstance(data_root, str):
            data_root = [data_root]
        if isinstance(data_root, list):
            self.data_root = dict()
            for dataset in DATASETS:
                self.data_root[dataset] = None
            for root in data_root:
                for dataset in DATASETS:
                    if dataset.lower() in root.lower():
                        self.data_root[dataset] = root
                        break
        if isinstance(data_root, dict):
            self.data_root = data_root

        self.verbose = verbose
        self.thickness = thickness

        if self.verbose:
            print('Dataset root')
            for dataset in DATASETS:
                print(dataset, ':', self.data_root[dataset])

        if self.verbose:
            print('Loading')
        self.metainfo = None
        data_list = []
        for file in self.ann_files:
            if isinstance(file, list):
                data_list += file
                continue
            elif isinstance(file, dict):
                if 'data_list' in file:
                    data = file
                else:
                    data_list.append(file)
                    continue
            elif isinstance(file, str):
                with open(file, 'rb') as f:
                    data = pickle.load(f)
            if self.metainfo is None:
                self.metainfo = data['metainfo']
            else:
                assert self.metainfo == data['metainfo']
            data_list += data['data_list']

        if isinstance(self.metainfo['categories'], list):
            self.classes = self.metainfo['categories']
            self.id_to_index = {i: i for i in range(len(self.classes))}
        elif isinstance(self.metainfo['categories'], dict):
            self.classes = list(self.metainfo['categories'].keys())
            self.id_to_index = {
                i: self.classes.index(classes)
                for classes, i in self.metainfo['categories'].items()
            }
        self.color_selector = ColorMap(classes=self.classes,
                                       init_file=color_setting)
        self.data = []
        for data in data_list:
            splits = data['sample_idx'].split('/')
            dataset = splits[0]
            data['dataset'] = dataset
            if self.data_root[dataset] is not None:
                if dataset == 'scannet':
                    region = splits[1]
                    dirpath = os.path.join(self.data_root['scannet'], 'scans',
                                           region)
                elif dataset == '3rscan':
                    region = splits[1]
                    dirpath = os.path.join(self.data_root['3rscan'], region)
                elif dataset == 'matterport3d':
                    building, region = splits[1], splits[2]
                    dirpath = os.path.join(self.data_root['matterport3d'],
                                           building)
                else:
                    region = splits[1]
                    dirpath = os.path.join(self.data_root[dataset], region)
                if os.path.exists(dirpath):
                    self.data.append(data)

        if self.verbose:
            print('Loading complete')

    def count_scenes(self):
        """Count the number of scenes."""
        return len(self.data)

    def list_categories(self):
        """List the categories involved in the dataset."""
        res = []
        for cate, id in self.metainfo['categories'].items():
            res.append({'category': cate, 'id': id})
        return res

    def list_scenes(self):
        """List all scenes in the dataset."""
        res = []
        for scene in self.data:
            res.append(scene['sample_idx'])
        return res

    def list_cameras(self, scene):
        """List all the camera frames in the scene.

        Args:
            scene (str): Scene name.

        Returns:
            list[str] or None: List of all the frame names. If there is no
            frames, we will return None.
        """
        for sample in self.data:
            if sample['sample_idx'] == scene:
                res = []
                dataset = sample['dataset']
                for img in sample['images']:
                    img_path = img['img_path']
                    if dataset == 'scannet':
                        cam_name = img_path.split('/')[-1][:-4]
                    elif dataset == '3rscan':
                        cam_name = img_path.split('/')[-1][:-10]
                    elif dataset == 'matterport3d':
                        cam_name = img_path.split(
                            '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                    else:
                        cam_name = img_path.split('/')[-1][:-4]
                    res.append(cam_name)
                return res

        print('No such scene')
        return None

    def list_instances(self, scene):
        """List all the instance annotations in the scene.

        Args:
            scene (str): Scene name.

        Returns:
            list[dict] or None: List of all the instance annotations. If there
            is no instances, we will return None.
        """
        for sample in self.data:
            if sample['sample_idx'] == scene:
                res = []
                for instance in sample['instances']:
                    label = self.classes[self.id_to_index[
                        instance['bbox_label_3d']]]
                    res.append({
                        '9dof_bbox': instance['bbox_3d'],
                        'label': label
                    })
                return res

        print('No such scene')
        return None

    def scene_info(self, scene_name):
        """Show the info of the given scene.

        Args:
            scene_name (str): Scene name.

        Returns:
            dict or None: Dict of scene info. If there is no such a scene, we
            will return None.
        """
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                if self.verbose:
                    print('Info of', scene_name)
                    print(len(scene['images']), 'images')
                    print(len(scene['instances']), 'boxes')
                return dict(num_images=len(scene['images']),
                            num_boxes=len(scene['instances']))

        if self.verbose:
            print('No such scene')
        return None

    def render_scene(self, scene_name, render_box=False):
        """Render a given scene with open3d.

        Args:
            scene_name (str): Scene name.
            render_box (bool): Whether to render the box in the scene.
                Defaults to False.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s
        select = None
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                select = scene
                break
        axis_align_matrix = select['axis_align_matrix']
        if dataset == 'scannet':
            filepath = os.path.join(self.data_root['scannet'], 'scans', region,
                                    f'{region}_vh_clean.ply')
        elif dataset == '3rscan':
            filepath = os.path.join(self.data_root['3rscan'], region,
                                    'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            filepath = os.path.join(self.data_root['matterport3d'], building,
                                    'region_segmentations', f'{region}.ply')
        else:
            raise NotImplementedError

        if self.verbose:
            print('Loading mesh')
        mesh = o3d.io.read_triangle_mesh(filepath, True)
        mesh.transform(axis_align_matrix)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if self.verbose:
            print('Loading complete')
        boxes = []
        if render_box:
            if self.verbose:
                print('Rendering box')
            for instance in select['instances']:
                box = _9dof_to_box(
                    instance['bbox_3d'],
                    self.classes[self.id_to_index[instance['bbox_label_3d']]],
                    self.color_selector)
                boxes += _box_add_thickness(box, self.thickness)
            if self.verbose:
                print('Rendering complete')
        o3d.visualization.draw_geometries([mesh, frame] + boxes)

    def render_continuous_scene(self,
                                scene_name,
                                start_cam=None,
                                pcd_downsample=100):
        """Render a scene with continuous ego-centric observations.

        Args:
            scene_name (str): Scene name.
            start_cam (str, optional): Camera frame from which the rendering
                starts. Defaults to None, corresponding to the first frame.
            pcd_downsample (int): The downsampling ratio of point clouds.
                Defaults to 100.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        selected_scene = None
        start_idx = -1
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                selected_scene = scene
                if start_cam is not None:
                    start_idx = -1
                    for i, img in enumerate(scene['images']):
                        img_path = img['img_path']
                        if dataset == 'scannet':
                            cam_name = img_path.split('/')[-1][:-4]
                        elif dataset == '3rscan':
                            cam_name = img_path.split('/')[-1][:-10]
                        elif dataset == 'matterport3d':
                            cam_name = img_path.split(
                                '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                        else:
                            cam_name = img_path.split('/')[-1][:-4]
                        if cam_name == start_cam:
                            start_idx = i
                            break
                    if start_idx == -1:
                        print('No such camera')
                        return
                else:
                    start_idx = 0

        if selected_scene is None:
            print('No such scene')
            return

        drawer = ContinuousDrawer(dataset, self.data_root[dataset],
                                  selected_scene, self.classes,
                                  self.id_to_index, self.color_selector,
                                  start_idx, pcd_downsample, self.thickness)
        drawer.begin()

    def render_continuous_occupancy(self, scene_name, start_cam=None):
        """Render occupancy with continuous ego-centric observations.

        Args:
            scene_name (str): Scene name.
            start_cam (str, optional): Camera frame from which the rendering
                starts. Defaults to None, corresponding to the first frame.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        selected_scene = None
        start_idx = -1
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                selected_scene = scene
                if start_cam is not None:
                    start_idx = -1
                    for i, img in enumerate(scene['images']):
                        img_path = img['img_path']
                        if dataset == 'scannet':
                            cam_name = img_path.split('/')[-1][:-4]
                        elif dataset == '3rscan':
                            cam_name = img_path.split('/')[-1][:-10]
                        elif dataset == 'matterport3d':
                            cam_name = img_path.split(
                                '/')[-1][:-8] + img_path.split('/')[-1][-7:-4]
                        else:
                            cam_name = img_path.split('/')[-1][:-4]
                        if cam_name == start_cam:
                            start_idx = i
                            break
                    if start_idx == -1:
                        print('No such camera')
                        return
                else:
                    start_idx = 0

        if selected_scene is None:
            print('No such scene')
            return

        drawer = ContinuousOccupancyDrawer(dataset, self.data_root[dataset],
                                           selected_scene, self.classes,
                                           self.id_to_index,
                                           self.color_selector, start_idx)
        drawer.begin()

    def render_occupancy(self, scene_name):
        """Render the occupancy annotation of a given scene.

        Args:
            scene_name (str): Scene name.
        """
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        if dataset == 'scannet':
            filepath = os.path.join(self.data_root['scannet'], 'scans', region,
                                    'occupancy', 'occupancy.npy')
        elif dataset == '3rscan':
            filepath = os.path.join(self.data_root['3rscan'], region,
                                    'occupancy', 'occupancy.npy')
        elif dataset == 'matterport3d':
            filepath = os.path.join(self.data_root['matterport3d'], building,
                                    'occupancy', f'occupancy_{region}.npy')
        else:
            raise NotImplementedError

        if self.verbose:
            print('Loading occupancy')
        gt_occ = np.load(filepath)
        if self.verbose:
            print('Loading complete')
        point_cloud_range = [-3.2, -3.2, -1.28 + 0.5, 3.2, 3.2, 1.28 + 0.5]
        # occ_size = [40, 40, 16]
        grid_size = [0.16, 0.16, 0.16]
        points = np.zeros((gt_occ.shape[0], 6), dtype=float)
        for i in range(gt_occ.shape[0]):
            x, y, z, label_id = gt_occ[i]
            label_id = int(label_id)
            label = 'object'
            if label_id == 0:
                label = 'object'
            else:
                label = self.classes[self.id_to_index[label_id]]
            color = self.color_selector.get_color(label)
            color = [x / 255.0 for x in color]
            points[i][:3] = [
                x * grid_size[0] + point_cloud_range[0] + grid_size[0] / 2,
                y * grid_size[1] + point_cloud_range[1] + grid_size[1] / 2,
                z * grid_size[2] + point_cloud_range[2] + grid_size[2] / 2
            ]
            points[i][3:] = color
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=grid_size[0])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([frame, voxel_grid])

    def show_image(self, scene_name, camera_name, render_box=False):
        """Render an ego-centric image view with annotations.

        Args:
            scene_name (str): Scene name.
            camera_name (str): The name of rendered camera frame.
            render_box (bool): Whether to render box annotations in the image.
                Defaults to False.
        """
        dataset = scene_name.split('/')[0]
        select = None
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                select = scene
        for camera in select['images']:
            img_path = camera['img_path']
            img_path = os.path.join(self.data_root[dataset],
                                    img_path[img_path.find('/') + 1:])
            if dataset == 'scannet':
                cam_name = img_path.split('/')[-1][:-4]
            elif dataset == '3rscan':
                cam_name = img_path.split('/')[-1][:-10]
            elif dataset == 'matterport3d':
                cam_name = img_path.split('/')[-1][:-8] + img_path.split(
                    '/')[-1][-7:-4]
            else:
                cam_name = img_path.split('/')[-1][:-4]
            if cam_name == camera_name:
                axis_align_matrix = select['axis_align_matrix']
                extrinsic = axis_align_matrix @ camera['cam2global']
                if 'cam2img' in camera:
                    intrinsic = camera['cam2img']
                else:
                    intrinsic = select['cam2img']
                img_drawer = ImageDrawer(img_path, verbose=self.verbose)
                if render_box:
                    if self.verbose:
                        print('Rendering box')
                    for i in camera['visible_instance_ids']:
                        instance = select['instances'][i]
                        box = _9dof_to_box(
                            instance['bbox_3d'], self.classes[self.id_to_index[
                                instance['bbox_label_3d']]],
                            self.color_selector)
                        label = self.classes[self.id_to_index[
                            instance['bbox_label_3d']]]
                        color = self.color_selector.get_color(label)
                        img_drawer.draw_box3d(box,
                                              color,
                                              label,
                                              extrinsic=extrinsic,
                                              intrinsic=intrinsic)
                    if self.verbose:
                        print('Rendering complete')

                img_drawer.show()
                return

        print('No such camera')
        return
