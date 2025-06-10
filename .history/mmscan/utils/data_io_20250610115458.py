import json
import os

import numpy as np
from tqdm import tqdm


def load_json(path: str):
    """Check the path and read the json file.

    Args:
        path (str): the path of the json file.
    Returns:
        the data in the json file.
    """
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def read_annotation_pickle(path: str, show_progress: bool = True):
    """Read annotation pickle file and return a dictionary, the embodiedscan
    annotation for all scans in the split.

    Args:
        path (str): the path of the annotation pickle file.
        show_progress (bool): whether showing the progress.
    Returns:
        dict: A dictionary.
            scene_id : (bboxes, object_ids, object_types,
                visible_view_object_dict, extrinsics_c2w,
                axis_align_matrix, intrinsics, image_paths)
            bboxes:
                numpy array of bounding boxes,
                shape (N, 9): xyz, lwh, ypr
            object_ids:
                numpy array of obj ids, shape (N,)
            object_types:
                list of strings, each string is a type of object
            visible_view_object_dict:
                a dictionary {view_id: visible_instance_ids}
            extrinsics_c2w:
                a list of 4x4 matrices, each matrix is the extrinsic
                matrix of a view
            axis_align_matrix:
                a 4x4 matrix, the axis-aligned matrix of the scene
            intrinsics:
                a list of 4x4 matrices, each matrix is the intrinsic
                matrix of a view
            image_paths:
                a list of strings, each string is the path of an image
                in the scene
    """
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)

    metainfo = data['metainfo']
    object_type_to_int = metainfo['categories']
    object_int_to_type = {v: k for k, v in object_type_to_int.items()}
    datalist = data['data_list']
    output_data = {}
    pbar = (tqdm(range(len(datalist))) if show_progress else range(
        len(datalist)))
    for scene_idx in pbar:
        images = datalist[scene_idx]['images']
        intrinsic = datalist[scene_idx].get('cam2img', None)  # a 4x4 matrix
        missing_intrinsic = False
        if intrinsic is None:
            missing_intrinsic = (
                True  # each view has different intrinsic for mp3d
            )
        depth_intrinsic = datalist[scene_idx].get(
            'cam2depth', None)  # a 4x4 matrix, for 3rscan
        if depth_intrinsic is None and not missing_intrinsic:
            depth_intrinsic = datalist[scene_idx][
                'depth_cam2img']  # a 4x4 matrix, for scannet
        axis_align_matrix = datalist[scene_idx][
            'axis_align_matrix']  # a 4x4 matrix

        scene_id = datalist[scene_idx]['sample_idx']
        if 'instances' in datalist[scene_idx]:
            instances = datalist[scene_idx]['instances']
            bboxes = []
            object_ids = []
            object_types = []
            object_type_ints = []
            for object_idx in range(len(instances)):
                bbox_3d = instances[object_idx]['bbox_3d']  # list of 9 values
                bbox_label_3d = instances[object_idx]['bbox_label_3d']  # int
                bbox_id = instances[object_idx]['bbox_id']  # int
                object_type = object_int_to_type[bbox_label_3d]

                object_type_ints.append(bbox_label_3d)
                object_types.append(object_type)
                bboxes.append(bbox_3d)
                object_ids.append(bbox_id)
            bboxes = np.array(bboxes)
            object_ids = np.array(object_ids)
            object_type_ints = np.array(object_type_ints)

        visible_view_object_dict = {}
        visible_view_object_list = []
        extrinsics_c2w = []
        intrinsics = []
        depth_intrinsics = []
        image_paths = []
        depth_image_paths = []

        for image_idx in range(len(images)):
            img_path = images[image_idx]['img_path']  # str
            depth_image = images[image_idx]['depth_path']
            extrinsic_id = img_path.split('/')[-1].split('.')[0]  # str
            cam2global = images[image_idx]['cam2global']  # a 4x4 matrix

            if missing_intrinsic:
                intrinsic = images[image_idx]['cam2img']

                depth_intrinsic = images[image_idx]['cam2img']
            if 'instances' in datalist[scene_idx]:
                visible_instance_indices = images[image_idx][
                    'visible_instance_ids']  # numpy array of int
                visible_instance_ids = object_ids[visible_instance_indices]
                visible_view_object_dict[extrinsic_id] = visible_instance_ids
                visible_view_object_list.append(visible_instance_ids)
            extrinsics_c2w.append(cam2global)
            intrinsics.append(intrinsic)
            depth_intrinsics.append(depth_intrinsic)
            image_paths.append(img_path)
            depth_image_paths.append(depth_image)
        if show_progress:
            pbar.set_description(f'Processing scene {scene_id}')
        extrinsics_c2w = [(axis_align_matrix @ extrinsic) for extrinsic in
                                                            extrinsics_c2w]
        output_data[scene_id] = {
            # image level
            'extrinsics_c2w': extrinsics_c2w,
            'axis_align_matrix': axis_align_matrix,
            'intrinsics': intrinsics,
            'depth_intrinsics': depth_intrinsics,
            'image_paths': image_paths,
            'depth_image_paths': depth_image_paths,
        }
        if 'instances' in datalist[scene_idx]:
            output_data[scene_id].update({
                # object level
                'bboxes':
                bboxes,
                'object_ids':
                object_ids,
                'object_types':
                object_types,
                'object_type_ints':
                object_type_ints,
                # image level
                'visible_instance_ids':
                visible_view_object_list,
                'visible_view_object_dict':
                visible_view_object_dict
            })
    return output_data


class id_mapping:
    """We rename the scan for consistency.

    This class is used to map the original scan names to the new names.
    Args:
        mp3d_mapping_path (str): the path of the mp3d mapping file.
    """

    def __init__(self, mp3d_mapping_path: str):

        def reverse_dict(mapping):
            re_mapping = {mapping[k]: k for k in mapping.keys()}
            return re_mapping

        with open(mp3d_mapping_path, 'r') as f:
            self.mp3d_mapping = json.load(f)

        self.mp3d_mapping_trans = reverse_dict(self.mp3d_mapping)

    def forward(self, scan_name: str):
        """map forward the original scan names to the new names.

        Args:
            scan_name (str): the original scan name.
        Returns:
            str: the new name.
        """

        if 'matterport3d/' in scan_name:
            scan_, region_ = (
                self.mp3d_mapping[scan_name.split('/')[1]],
                scan_name.split('/')[2],
            )
            return scan_ + '_' + region_
        elif '3rscan' in scan_name:
            return scan_name.split('/')[1]
        elif 'scannet' in scan_name:
            return scan_name.split('/')[1]
        else:
            raise ValueError(f'{scan_name} is not a scan name')

    def backward(self, scan_name: str):
        """map backward the new names to the original scan names.

        Args:
            scan_name (str): the new name.
        Returns:
            str: the original scan name.
        """
        if '1mp3d' in scan_name:
            scene1, scene2, region = scan_name.split('_')
            return ('matterport3d/' +
                    self.mp3d_mapping_trans[scene1 + '_' + scene2] + '/' +
                    region)
        elif '3rscan' in scan_name:
            return '3rscan/' + scan_name
        elif 'scene' in scan_name:
            return 'scannet/' + scan_name
        else:
            raise ValueError(f'{scan_name} is not a scan name')
