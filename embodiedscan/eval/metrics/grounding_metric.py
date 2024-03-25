# Copyright (c) OpenRobotLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence

import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from embodiedscan.registry import METRICS
from embodiedscan.structures import EulerDepthInstance3DBoxes


@METRICS.register_module()
class GroundingMetric(BaseMetric):
    """Lanuage grounding evaluation metric. We calculate the grounding
    performance based on the alignment score of each bbox with the input
    prompt.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Whether to only inference the predictions without
            evaluation. Defaults to False.
        result_dir (str): Dir to save results, e.g., if result_dir = './',
            the result file will be './test_results.json'. Defaults to ''.
    """

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only=False,
                 result_dir='') -> None:
        super(GroundingMetric, self).__init__(prefix=prefix,
                                              collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.prefix = prefix
        self.format_only = format_only
        self.result_dir = result_dir

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def ground_eval(self, gt_annos, det_annos, logger=None):

        assert len(det_annos) == len(gt_annos)

        pred = {}
        gt = {}

        object_types = [
            'Easy', 'Hard', 'View-Dep', 'View-Indep', 'Unique', 'Multi',
            'Overall'
        ]

        for t in self.iou_thr:
            for object_type in object_types:
                pred.update({object_type + '@' + str(t): 0})
                gt.update({object_type + '@' + str(t): 1e-14})

        for sample_id in range(len(det_annos)):
            det_anno = det_annos[sample_id]
            gt_anno = gt_annos[sample_id]
            target_scores = det_anno['target_scores_3d']  # (num_query, )

            bboxes = det_anno['bboxes_3d']
            gt_bboxes = gt_anno['gt_bboxes_3d']
            bboxes = EulerDepthInstance3DBoxes(bboxes.tensor,
                                               origin=(0.5, 0.5, 0.5))
            gt_bboxes = EulerDepthInstance3DBoxes(gt_bboxes.tensor,
                                                  origin=(0.5, 0.5, 0.5))

            view_dep = gt_anno['is_view_dep']
            hard = gt_anno['is_hard']
            unique = gt_anno['is_unique']

            box_index = target_scores.argsort(dim=-1, descending=True)[:10]
            top_bbox = bboxes[box_index]

            iou = top_bbox.overlaps(top_bbox, gt_bboxes)  # (num_query, 1)

            for t in self.iou_thr:
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

        for t in self.iou_thr:
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
            print_log('\n' + table.table, logger=logger)

        return ret_dict

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results after all batches have
        been processed.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()  # noqa
        annotations, preds = zip(*results)
        ret_dict = {}
        if self.format_only:
            # preds is a list of dict
            results = []
            for pred in preds:
                result = dict()
                # convert the Euler boxes to the numpy array to save
                bboxes_3d = pred['bboxes_3d'].tensor
                scores_3d = pred['scores_3d']
                # Note: hard-code save top-20 predictions
                # eval top-10 predictions during the test phase by default
                box_index = scores_3d.argsort(dim=-1, descending=True)[:20]
                top_bboxes_3d = bboxes_3d[box_index]
                top_scores_3d = scores_3d[box_index]
                result['bboxes_3d'] = top_bboxes_3d.numpy()
                result['scores_3d'] = top_scores_3d.numpy()
                results.append(result)
            mmengine.dump(results,
                          os.path.join(self.result_dir, 'test_results.json'))
            return ret_dict

        ret_dict = self.ground_eval(annotations, preds)

        return ret_dict
