import os
from typing import List, Union

import mmengine
import numpy as np
import open3d as o3d
from utils.color_selector import ColorMap
from utils.img_drawer import ImageDrawer

DATASETS = ['scannet', '3rscan', 'matterport3d']


class EmbodiedScanExplorer:

    def __init__(
        self,
        dataroot: Union[dict, List],
        ann_file: Union[dict, List, str],
        verbose: bool = False,
        color_setting: str = None,
    ):

        if isinstance(ann_file, dict):
            ann_file = list(ann_file.values())
        elif isinstance(ann_file, str):
            ann_file = [ann_file]
        self.ann_files = ann_file

        if isinstance(dataroot, str):
            dataroot = [dataroot]
        if isinstance(dataroot, list):
            self.dataroot = dict()
            for dataset in DATASETS:
                self.dataroot[dataset] = None
            for root in dataroot:
                for dataset in DATASETS:
                    if dataset.lower() in root.lower():
                        self.dataroot[dataset] = root
                        break
        if isinstance(dataroot, dict):
            self.dataroot = dataroot

        self.verbose = verbose

        if self.verbose:
            print('Dataset root')
            for dataset in DATASETS:
                print(dataset, ':', self.dataroot[dataset])

        if self.verbose:
            print('Loading')
        self.metainfo = None
        data_list = []
        for file in self.ann_files:
            data = mmengine.load(file)
            if self.metainfo is None:
                self.metainfo = data['metainfo']
            else:
                assert self.metainfo == data['metainfo']
            data_list += data['data_list']

        self.classes = list(self.metainfo['categories'].keys())
        self.color_selector = ColorMap(classes=self.classes,
                                       init_file=color_setting)
        self.data = []
        for data in data_list:
            splits = data['sample_idx'].split('/')
            data['dataset'] = splits[0]
            if self.dataroot[splits[0]] is not None:
                self.data.append(data)

        if self.verbose:
            print('Loading complete')

    def count_scenes(self):
        return len(self.data)

    def list_scenes(self):
        res = []
        for scene in self.data:
            res.append(scene['sample_idx'])
        return res

    def scene_info(self, scene_name):
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
            filepath = os.path.join(self.dataroot['scannet'], 'scans', region,
                                    f'{region}_vh_clean.ply')
        elif dataset == '3rscan':
            filepath = os.path.join(self.dataroot['3rscan'], region,
                                    'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            filepath = os.path.join(self.dataroot['matterport3d'], building,
                                    'region_segmentations', f'{region}.ply')
        else:
            raise NotImplementedError

        mesh = o3d.io.read_triangle_mesh(filepath, True)
        mesh.transform(axis_align_matrix)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        boxes = []
        if render_box:
            for instance in select['instances']:
                box = self._9dof_to_box(instance['bbox_3d'],
                                        instance['bbox_label_3d'])
                boxes.append(box)
        o3d.visualization.draw_geometries([mesh, frame] + boxes)

    def render_occupancy(self, scene_name):
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s

        if dataset == 'scannet':
            filepath = os.path.join(self.dataroot['scannet'], 'scans', region,
                                    'occupancy', 'occupancy.npy')
        elif dataset == '3rscan':
            filepath = os.path.join(self.dataroot['3rscan'], region,
                                    'occupancy', 'occupancy.npy')
        elif dataset == 'matterport3d':
            filepath = os.path.join(self.dataroot['matterport3d'], building,
                                    'occupancy', f'occupancy_{region}.npy')
        else:
            raise NotImplementedError

        gt_occ = np.load(filepath)
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
                label = self.classes[label_id - 1]
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

    def render_image(self, scene_name, camera_name):
        dataset = scene_name.split('/')[0]
        select = None
        for scene in self.data:
            if scene['sample_idx'] == scene_name:
                select = scene
        for camera in select['images']:
            img_path = camera['img_path']
            img_path = os.path.join(self.dataroot[dataset],
                                    img_path[img_path.find('/') + 1:])
            if dataset == 'scannet':
                cam_name = img_path.split('/')[-1][:-4]
            elif dataset == '3rscan':
                cam_name = img_path.split('/')[-1][:-10]
            elif dataset == 'matterport3d':
                cam_name = img_path.split('/')[-1][:-8] + img_path.split(
                    '/')[-1][-7:-4]
            if cam_name == camera_name:
                axis_align_matrix = select['axis_align_matrix']
                extrinsic = axis_align_matrix @ camera['cam2global']
                if 'cam2img' in camera:
                    intrinsic = camera['cam2img']
                else:
                    intrinsic = select['cam2img']
                img_drawer = ImageDrawer(img_path, verbose=self.verbose)
                for i in camera['visible_instance_ids']:
                    instance = select['instances'][i]
                    box = self._9dof_to_box(instance['bbox_3d'],
                                            instance['bbox_label_3d'])
                    label = self.classes[instance['bbox_label_3d'] - 1]
                    color = self.color_selector.get_color(label)
                    img_drawer.draw_box3d(box,
                                          color,
                                          label,
                                          extrinsic=extrinsic,
                                          intrinsic=intrinsic)

                img_drawer.show()
                return

        print('No such camera')
        return

    def _9dof_to_box(self, box, label_id):
        if isinstance(box, list):
            box = np.array(box)
        center = box[:3].reshape(3, 1)
        scale = box[3:6].reshape(3, 1)
        rot = box[6:].reshape(3, 1)
        rot_mat = \
            o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
        geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

        label = self.classes[label_id - 1]
        color = self.color_selector.get_color(label)
        color = [x / 255.0 for x in color]
        geo.color = color
        return geo


if __name__ == '__main__':
    a = EmbodiedScanExplorer(
        dataroot=['data/scannet', 'data/3rscan/', 'data/matterport3d/'],
        ann_file=[
            'data/full_10_visible/embodiedscan_infos_train_full.pkl',
            'data/full_10_visible/embodiedscan_infos_val_full.pkl'
        ],
        verbose=True)
    print(a.list_scenes())
    print(a.count_scenes())
    a.render_image('scannet/scene0000_00', '00000')
