# Copyright (c) OpenRobotLab. All rights reserved.
import argparse

import mmengine
from mmengine.logging import print_log
from terminaltables import AsciiTable

from embodiedscan.structures import EulerDepthInstance3DBoxes


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('results_file', help='the results pkl file')
    parser.add_argument('ann_file', help='annoations json file')

    parser.add_argument('--iou_thr',
                        type=list,
                        default=[0.25, 0.5],
                        help='the IoU threshold during evaluation')

    args = parser.parse_args()
    return args


def ground_eval(gt_annos, det_annos, iou_thr):

    assert len(det_annos) == len(gt_annos)

    pred = {}
    gt = {}

    object_types = [
        'Easy', 'Hard', 'View-Dep', 'View-Indep', 'Unique', 'Multi', 'Overall'
    ]

    for t in iou_thr:
        for object_type in object_types:
            pred.update({object_type + '@' + str(t): 0})
            gt.update({object_type + '@' + str(t): 1e-14})

    for sample_id in range(len(det_annos)):
        det_anno = det_annos[sample_id]
        gt_anno = gt_annos[sample_id]['ann_info']

        bboxes = det_anno['bboxes_3d']
        gt_bboxes = gt_anno['gt_bboxes_3d']
        bboxes = EulerDepthInstance3DBoxes(bboxes, origin=(0.5, 0.5, 0.5))
        gt_bboxes = EulerDepthInstance3DBoxes(gt_bboxes,
                                              origin=(0.5, 0.5, 0.5))
        scores = bboxes.tensor.new_tensor(
            det_anno['scores_3d'])  # (num_query, )

        view_dep = gt_anno['is_view_dep']
        hard = gt_anno['is_hard']
        unique = gt_anno['is_unique']

        box_index = scores.argsort(dim=-1, descending=True)[:10]
        top_bboxes = bboxes[box_index]

        iou = top_bboxes.overlaps(top_bboxes, gt_bboxes)  # (num_query, 1)

        for t in iou_thr:
            threshold = iou > t
            found = int(threshold.any())
            if view_dep:
                gt['View-Dep@' + str(t)] += 1
                pred['View-Dep@' + str(t)] += found
            else:
                gt['View-Indep@' + str(t)] += 1
                pred['View-Indep@' + str(t)] += found
            if hard:
                gt['Hard@' + str(t)] += 1
                pred['Hard@' + str(t)] += found
            else:
                gt['Easy@' + str(t)] += 1
                pred['Easy@' + str(t)] += found
            if unique:
                gt['Unique@' + str(t)] += 1
                pred['Unique@' + str(t)] += found
            else:
                gt['Multi@' + str(t)] += 1
                pred['Multi@' + str(t)] += found

            gt['Overall@' + str(t)] += 1
            pred['Overall@' + str(t)] += found

    header = ['Type']
    header.extend(object_types)
    ret_dict = {}

    for t in iou_thr:
        table_columns = [['results']]
        for object_type in object_types:
            metric = object_type + '@' + str(t)
            value = pred[metric] / max(gt[metric], 1)
            ret_dict[metric] = value
            table_columns.append([f'{value:.4f}'])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table)

    return ret_dict


def main():
    args = parse_args()
    preds = mmengine.load(args.results_file)['results']
    annotations = mmengine.load(args.ann_file)
    assert len(preds) == len(annotations)
    ground_eval(annotations, preds, args.iou_thr)


if __name__ == '__main__':
    main()
