import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.config import CONF
from utils.box_util import box3d_iou

SCANREFER_GT = json.load(
    open(os.path.join(CONF.PATH.DATA, 'ScanRefer_filtered_test_gt_bbox.json')))


def organize_gt():
    organized = {}

    for data in SCANREFER_GT:
        scene_id = data['scene_id']
        object_id = data['object_id']
        ann_id = data['ann_id']

        if scene_id not in organized:
            organized[scene_id] = {}

        if object_id not in organized[scene_id]:
            organized[scene_id][object_id] = {}

        if ann_id not in organized[scene_id][object_id]:
            organized[scene_id][object_id][ann_id] = {}

        organized[scene_id][object_id][ann_id] = data

    return organized


def evaluate(args):
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, 'pred.json')
    if not os.path.isfile(pred_path):
        print(
            'please run `benchmark/predict.py` first to generate bounding boxes'
        )
        exit()

    organized_gt = organize_gt()

    with open(pred_path) as f:
        predictions = json.load(f)
        ious = []
        masks = []
        others = []
        print('evaluating...')
        for data in tqdm(predictions):
            scene_id = data['scene_id']
            object_id = data['object_id']
            ann_id = data['ann_id']
            pred_bbox = np.array(data['bbox'])
            mask = data['unique_multiple']
            other = data['others']

            try:
                gt_bbox = np.array(
                    organized_gt[scene_id][object_id][ann_id]['bbox'])
                # iou, _ = box3d_iou(pred_bbox, gt_bbox)
                iou = box3d_iou(pred_bbox, gt_bbox)

            except KeyError:
                iou = 0

            ious.append(iou)
            masks.append(mask)
            others.append(other)

        # ious = np.array(ious)
        # iou_rate_025 = ious[ious >= 0.25].shape[0] / ious.shape[0]
        # iou_rate_05 = ious[ious >= 0.5].shape[0] / ious.shape[0]

        # print("\nAcc@0.25IoU: {}".format(iou_rate_025))
        # print("Acc@0.5IoU: {}".format(iou_rate_05))

        ious = np.array(ious)
        masks = np.array(masks)
        others = np.array(others)

        multiple_dict = {'unique': 0, 'multiple': 1}
        others_dict = {'not_in_others': 0, 'in_others': 1}

        # evaluation stats
        stats = {k: np.sum(masks == v) for k, v in multiple_dict.items()}
        stats['overall'] = masks.shape[0]
        stats = {}
        for k, v in multiple_dict.items():
            stats[k] = {}
            for k_o, v_o in others_dict.items():
                stats[k][k_o] = np.sum(
                    np.logical_and(masks == v, others == v_o))

            stats[k]['overall'] = np.sum(masks == v)

        stats['overall'] = {}
        for k_o, v_o in others_dict.items():
            stats['overall'][k_o] = np.sum(others == v_o)

        stats['overall']['overall'] = masks.shape[0]

        # aggregate scores
        scores = {}
        for k, v in multiple_dict.items():
            for k_o in others_dict.keys():
                acc_025iou = ious[np.logical_and(np.logical_and(masks == multiple_dict[k], others == others_dict[k_o]), ious >= 0.25)].shape[0] \
                    / ious[np.logical_and(masks == multiple_dict[k], others == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks == multiple_dict[k], others == others_dict[k_o])) > 0 else 0
                acc_05iou = ious[np.logical_and(np.logical_and(masks == multiple_dict[k], others == others_dict[k_o]), ious >= 0.5)].shape[0] \
                    / ious[np.logical_and(masks == multiple_dict[k], others == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks == multiple_dict[k], others == others_dict[k_o])) > 0 else 0

                if k not in scores:
                    scores[k] = {k_o: {} for k_o in others_dict.keys()}

                scores[k][k_o]['acc@0.25iou'] = acc_025iou
                scores[k][k_o]['acc@0.5iou'] = acc_05iou

            acc_025iou = ious[np.logical_and(masks == multiple_dict[k], ious >= 0.25)].shape[0] \
                / ious[masks == multiple_dict[k]].shape[0] if np.sum(masks == multiple_dict[k]) > 0 else 0
            acc_05iou = ious[np.logical_and(masks == multiple_dict[k], ious >= 0.5)].shape[0] \
                / ious[masks == multiple_dict[k]].shape[0] if np.sum(masks == multiple_dict[k]) > 0 else 0

            scores[k]['overall'] = {}
            scores[k]['overall']['acc@0.25iou'] = acc_025iou
            scores[k]['overall']['acc@0.5iou'] = acc_05iou

        scores['overall'] = {}
        for k_o in others_dict.keys():
            acc_025iou = ious[np.logical_and(others == others_dict[k_o], ious >= 0.25)].shape[0] \
                / ious[others == others_dict[k_o]].shape[0] if np.sum(others == others_dict[k_o]) > 0 else 0
            acc_05iou = ious[np.logical_and(others == others_dict[k_o], ious >= 0.5)].shape[0] \
                / ious[others == others_dict[k_o]].shape[0] if np.sum(others == others_dict[k_o]) > 0 else 0

            # aggregate
            scores['overall'][k_o] = {}
            scores['overall'][k_o]['acc@0.25iou'] = acc_025iou
            scores['overall'][k_o]['acc@0.5iou'] = acc_05iou

        acc_025iou = ious[ious >= 0.25].shape[0] / ious.shape[0]
        acc_05iou = ious[ious >= 0.5].shape[0] / ious.shape[0]

        # aggregate
        scores['overall']['overall'] = {}
        scores['overall']['overall']['acc@0.25iou'] = acc_025iou
        scores['overall']['overall']['acc@0.5iou'] = acc_05iou

        # report
        print('\nstats:')
        for k_s in stats.keys():
            for k_o in stats[k_s].keys():
                print('{} | {}: {}'.format(k_s, k_o, stats[k_s][k_o]))

        for k_s in scores.keys():
            print('\n{}:'.format(k_s))
            for k_m in scores[k_s].keys():
                for metric in scores[k_s][k_m].keys():
                    print('{} | {} | {}: {}'.format(k_s, k_m, metric,
                                                    scores[k_s][k_m][metric]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        type=str,
                        help='Folder containing the model')
    args = parser.parse_args()

    evaluate(args)
