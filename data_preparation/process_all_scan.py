import json
import os
from argparse import ArgumentParser

import mmengine
import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix
from tqdm import tqdm
from utils.data_utils import read_annotation_pickle
from utils.mp3d_process import process_mp3d
from utils.pcd_utils import is_inside_box
from utils.proc_utils import mmengine_track_func
from utils.scannet_process import process_scannet
from utils.trscan_process import process_trscan

dict_1 = {}


def create_scene_pcd(es_anno, pcd_result):
    """Adding the embodiedscan-box annotation into the point clouds data.

    Args:
        es_anno (dict): The embodiedscan annotation of
            the target scan.
        pcd_result (tuple) :
            (1) aliged point clouds coordinates
                shape (n,3)
            (2) point clouds color ([0,1])
                shape (n,3)
            (3) label (no need here)

    Returns:
        tuple :
            (1) aliged point clouds coordinates
                shape (n,3)
            (2) point clouds color ([0,1])
                shape (n,3)
            (3) point clouds label (int)
                shape (n,1)
            (4) point clouds object id (int)
                shape (n,1)
    """
    pc, color, label = pcd_result
    label = np.ones_like(label) * -100
    instance_ids = np.ones(pc.shape[0], dtype=np.int16) * (-100)
    bboxes = es_anno['bboxes'].reshape(-1, 9)
    bboxes[:, 3:6] = np.clip(bboxes[:, 3:6], a_min=1e-2, a_max=None)
    object_ids = es_anno['object_ids']
    object_types = es_anno['object_types']  # str
    sorted_indices = sorted(enumerate(bboxes),
                            key=lambda x: -np.prod(x[1][3:6]))
    # the larger the box, the smaller the index
    sorted_indices_list = [index for index, value in sorted_indices]

    bboxes = [bboxes[index] for index in sorted_indices_list]
    object_ids = [object_ids[index] for index in sorted_indices_list]
    object_types = [object_types[index] for index in sorted_indices_list]

    for box, obj_id, obj_type in zip(bboxes, object_ids, object_types):
        obj_type_id = TYPE2INT.get(obj_type, -1)
        center, size = box[:3], box[3:6]

        orientation = np.array(
            euler_angles_to_matrix(torch.tensor(box[np.newaxis, 6:]),
                                   convention='ZXY')[0])

        box_pc_mask = is_inside_box(pc, center, size, orientation)

        instance_ids[box_pc_mask] = obj_id
        label[box_pc_mask] = obj_type_id
    return pc, color, label, instance_ids


@mmengine_track_func
def process_one_scan(
    scan_id,
    save_root,
    scannet_root,
    mp3d_root,
    trscan_root,
    scannet_matrix,
    mp3d_matrix,
    trscan_matrix,
    mp3d_mapping,
):
    """Process the point clouds of one scan and save in a pth file.

    The pth file is a tuple of:
        (1) aliged point clouds coordinates
            shape (n,3)
        (2) point clouds color ([0,1])
            shape (n,3)
        (3) point clouds label (int)
            shape (n,1)
        (4) point clouds object id (int)
            shape (n,1)
    Args:
        scan_id (str): the scan id
    """

    if os.path.exists(f'{save_root}/{scan_id}.pth'):
        return

    try:
        if 'scene' in scan_id:
            if 'scannet/' + scan_id not in dict_1:
                return

            pcd_info = create_scene_pcd(
                dict_1['scannet/' + scan_id],
                process_scannet(scan_id, scannet_root, scannet_matrix),
            )

        elif 'mp3d' in scan_id:
            raw_scan_id, region_id = (
                mp3d_mapping[scan_id.split('_region')[0]],
                'region' + scan_id.split('_region')[1],
            )
            mapping_name = f'matterport3d/{raw_scan_id}/{region_id}'
            if mapping_name not in dict_1:
                return

            pcd_info = create_scene_pcd(
                dict_1[mapping_name],
                process_mp3d(scan_id, mp3d_root, mp3d_matrix, mp3d_mapping),
            )

        else:
            if '3rscan/' + scan_id not in dict_1:
                return
            pcd_info = create_scene_pcd(
                dict_1['3rscan/' + scan_id],
                process_trscan(scan_id, trscan_root, trscan_matrix),
            )

        save_path = f'{save_root}/{scan_id}.pth'
        torch.save(pcd_info, save_path)

    except Exception as error:
        print(error)
        print(f'Error in processing {scan_id}')


if __name__ == '__main__':
    path_of_version1 = '../mmscan_data/embodiedscan-split/embodiedscan-v1'
    parser = ArgumentParser()
    parser.add_argument('--meta_path', type=str, default='./meta-data')
    parser.add_argument(
        '--data_root',
        type=str,
        default=f'{path_of_version1}/data',
    )
    parser.add_argument(
        '--save_root',
        type=str,
        default=f'{path_of_version1}/process_pcd',
    )
    parser.add_argument(
        '--train_pkl_path',
        type=str,
        default=f'{path_of_version1}/embodiedscan_infos_train.pkl',
    )
    parser.add_argument(
        '--val_pkl_path',
        type=str,
        default=f'{path_of_version1}/embodiedscan_infos_val.pkl',
    )
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    scannet_root = f'{args.data_root}/scannet'
    mp3d_root = f'{args.data_root}/matterport3d'
    trscan_root = f'{args.data_root}/3rscan'

    # (0) some necessary info
    with open(f'{args.meta_path}/mp3d_mapping.json', 'r') as f:
        mapping = json.load(f)
    mapping = {v: k for k, v in mapping.items()}

    TYPE2INT = np.load(args.train_pkl_path,
                       allow_pickle=True)['metainfo']['categories']
    dict_1.update(read_annotation_pickle(args.train_pkl_path))
    dict_1.update(read_annotation_pickle(args.val_pkl_path))

    # loading the required scan id
    with open(f'{args.meta_path}/all_scan.json', 'r') as f:
        scan_id_list = json.load(f)

    # (1) loading the axis matrix info
    mp3d_matrix = np.load(f'{args.meta_path}/mp3d_matrix.npy',
                          allow_pickle=True).item()
    trscan_matrix = np.load(f'{args.meta_path}/3rscan_matrix.npy',
                            allow_pickle=True).item()
    with open(f'{args.meta_path}/scans_axis_alignment_matrices.json',
              'r') as f:
        scan2axis_align = json.load(f)
    scannet_matrix = {}
    for scan_id in scan2axis_align:
        scannet_matrix[scan_id] = np.array(scan2axis_align[scan_id],
                                           dtype=np.float32).reshape(4, 4)

    # (2) Collecting task
    tasks = []
    for scan_id in scan_id_list:
        tasks.append((
            scan_id,
            args.save_root,
            scannet_root,
            mp3d_root,
            trscan_root,
            scannet_matrix,
            mp3d_matrix,
            trscan_matrix,
            mapping,
        ))

    # (3) processing steps

    parallel = args.nproc > 1

    if parallel:
        mmengine.utils.track_parallel_progress(process_one_scan, tasks,
                                               args.nproc)
    else:
        for param in tqdm(tasks):
            process_one_scan(param)
