import argparse
import importlib
import json
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.ap_helper import APCalculator, parse_groundtruths, parse_predictions
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.eval_helper import get_eval
from lib.loss_helper import get_loss
from lib.solver import Solver
from models.refnet import RefNet
from utils.box_util import get_3d_box

SCANREFER_TEST = json.load(
    open(os.path.join(CONF.PATH.DATA, 'ScanRefer_filtered_test.json')))


def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(scanrefer=scanrefer,
                                      scanrefer_all_scene=all_scene_list,
                                      split=split,
                                      num_points=args.num_points,
                                      use_color=args.use_color,
                                      use_height=(not args.no_height),
                                      use_normal=args.use_normal,
                                      use_multiview=args.use_multiview)
    print('predict for {} samples'.format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, dataloader


def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(
        args.use_normal) * 3 + int(
            args.use_color) * 3 + int(not args.no_height)
    model = RefNet(num_class=config.num_class,
                   num_heading_bin=config.num_heading_bin,
                   num_size_cluster=config.num_size_cluster,
                   mean_size_arr=config.mean_size_arr,
                   num_proposal=args.num_proposals,
                   input_feature_dim=input_channels,
                   use_lang_classifier=(not args.no_lang_cls),
                   use_bidir=args.use_bidir).cuda()

    model_name = 'model.pth'
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model


def get_scannet_scene_list(split):
    scene_list = sorted([
        line.rstrip() for line in open(
            os.path.join(CONF.PATH.SCANNET_META, 'scannetv2_{}.txt'.format(
                split)))
    ])

    return scene_list


def get_scanrefer(args):
    scanrefer = SCANREFER_TEST
    scene_list = sorted(list(set([data['scene_id'] for data in scanrefer])))
    scanrefer = [data for data in scanrefer if data['scene_id'] in scene_list]

    return scanrefer, scene_list


def predict(args):
    print('predict bounding boxes...')
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print('preparing data...')
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, 'test', DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        'remove_empty_box': True,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': True,
        'per_class_proposal': True,
        'conf_thresh': 0.05,
        'dataset_config': DC
    } if not args.no_nms else None

    # predict
    print('predicting...')
    pred_bboxes = []
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        # feed
        data_dict = model(data_dict)
        _, data_dict = get_loss(data_dict=data_dict,
                                config=DC,
                                detection=False,
                                reference=True)

        objectness_preds_batch = torch.argmax(data_dict['objectness_scores'],
                                              2).long()

        if POST_DICT:
            _ = parse_predictions(data_dict, POST_DICT)
            nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

        pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks,
                                1)  # (B,)
        pred_center = data_dict['center']  # (B,K,3)

        pred_rot_mat = data_dict['rot_mat']
        # pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        # pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        # pred_heading_class = pred_heading_class # B,num_proposal
        # pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal

        pred_size_class = torch.argmax(data_dict['size_scores'],
                                       -1)  # B,num_proposal
        pred_size_residual = torch.gather(
            data_dict['size_residuals'], 2,
            pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, 1, 3))  # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2)  # B,num_proposal,3

        for i in range(pred_ref.shape[0]):
            # compute the iou
            pred_ref_idx = pred_ref[i]
            pred_obb = DC.param2obb(  # TODO yesname
                pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(),
                pred_rot_mat[i, pred_ref_idx].detach().cpu().numpy(),
                # pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(),
                # pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                pred_size_class[i, pred_ref_idx].detach().cpu().numpy(),
                pred_size_residual[i, pred_ref_idx].detach().cpu().numpy())
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])

            # construct the multiple mask
            multiple = data_dict['unique_multiple'][i].item()

            # construct the others mask
            others = 1 if data_dict['object_cat'][i] == 17 else 0

            # store data
            scanrefer_idx = data_dict['scan_idx'][i].item()
            pred_data = {
                'scene_id': scanrefer[scanrefer_idx]['scene_id'],
                'object_id': scanrefer[scanrefer_idx]['object_id'],
                'ann_id': scanrefer[scanrefer_idx]['ann_id'],
                'bbox': pred_bbox.tolist(),
                'unique_multiple': multiple,
                'others': others
            }
            pred_bboxes.append(pred_data)

    # dump
    print('dumping...')
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, 'pred.json')
    with open(pred_path, 'w') as f:
        json.dump(pred_bboxes, f, indent=4)

    print('done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        type=str,
                        help='Folder containing the model')
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--num_points',
                        type=int,
                        default=40000,
                        help='Point Number [default: 40000]')
    parser.add_argument('--num_proposals',
                        type=int,
                        default=256,
                        help='Proposal number [default: 256]')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no_height',
                        action='store_true',
                        help='Do NOT use height signal in input.')
    parser.add_argument('--no_lang_cls',
                        action='store_true',
                        help='Do NOT use language classifier.')
    parser.add_argument(
        '--no_nms',
        action='store_true',
        help='do NOT use non-maximum suppression for post-processing.')
    parser.add_argument('--use_color',
                        action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_normal',
                        action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_multiview',
                        action='store_true',
                        help='Use multiview images.')
    parser.add_argument('--use_bidir',
                        action='store_true',
                        help='Use bi-directional GRU.')
    args = parser.parse_args()

    # setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    predict(args)
