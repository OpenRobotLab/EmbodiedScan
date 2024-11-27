# Copyright (c) OpenRobotLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence

import mmengine
import numpy as np
import torch
from embodiedscan.registry import METRICS
from embodiedscan.structures import EulerDepthInstance3DBoxes
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from scipy.optimize import linear_sum_assignment
from terminaltables import AsciiTable
from tqdm import tqdm

from mmscan import VG_Evaluator


def abbr(sub_class):
    sub_class = sub_class.lower()
    sub_class = sub_class.replace('single', 'sngl')
    sub_class = sub_class.replace('inter', 'int')
    sub_class = sub_class.replace('unique', 'uniq')
    sub_class = sub_class.replace('common', 'cmn')
    sub_class = sub_class.replace('attribute', 'attr')
    if 'sngl' in sub_class and ('attr' in sub_class or 'eq' in sub_class):
        sub_class = 'vg_sngl_attr'
    return sub_class


@METRICS.register_module()
class GroundingMetricMod(BaseMetric):
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
        super(GroundingMetricMod, self).__init__(prefix=prefix,
                                                 collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.prefix = prefix
        self.format_only = format_only
        self.result_dir = result_dir
        self.mmscan_eval = VG_Evaluator(True)

    def to_mmscan_form(self, det_annos, gt_annos):
        batch_input = []
        for i, (gt_anno, det_anno) in tqdm(enumerate(zip(gt_annos,
                                                         det_annos))):

            _input = {}
            _input['pred_scores'] = det_anno['target_scores_3d']
            _input['pred_bboxes'] = det_anno['bboxes_3d']

            _input['gt_bboxes'] = gt_anno['gt_bboxes_3d']
            _input['subclass'] = gt_anno['sub_class']
            _input['pred_bboxes'] = torch.stack([euler_box for euler_box in _input['pred_bboxes']])\
                if len(_input['pred_bboxes']) > 0 else torch.empty(0, 9)

            _input['gt_bboxes'] = torch.stack([euler_box for euler_box in _input['gt_bboxes']]) \
                if len(_input['gt_bboxes']) > 0 else torch.empty(0, 9)

            batch_input.append(_input)

        return batch_input

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

        metric_results = {}

        batch_input = self.to_mmscan_form(det_annos, gt_annos)

        self.mmscan_eval.reset()

        self.mmscan_eval.update(batch_input)

        print('Staring evaluation!')
        self.mmscan_eval.start_evaluation()

        result_table = self.mmscan_eval.print_result()

        print_log('\n' + result_table, logger=logger)
        return metric_results

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
        # if self.format_only:
        #     # preds is a list of dict
        #     results = []
        #     print_log("If you see this, you are in function: compute metrics with format only.")
        #     for pred in preds:
        #         result = dict()
        #         # convert the Euler boxes to the numpy array to save
        #         bboxes_3d = pred['bboxes_3d'].tensor
        #         scores_3d = pred['scores_3d']
        #         # Note: hard-code save top-20 predictions
        #         # eval top-10 predictions during the test phase by default
        #         box_index = scores_3d.argsort(dim=-1, descending=True)
        #         top_bboxes_3d = bboxes_3d[box_index]
        #         top_scores_3d = scores_3d[box_index]
        #         result['bboxes_3d'] = top_bboxes_3d.numpy()
        #         result['scores_3d'] = top_scores_3d.numpy()
        #         results.append(result)
        #     mmengine.dump(results,
        #                   os.path.join(self.result_dir, 'test_results.json'))
        #     return ret_dict
        # try:
        #     torch.save({"pred_list":preds,"gt_list":annotations},"/mnt/petrelfs/linjingli/tmp/data/big_tmp/result_og_es2.pt")
        # except:
        #     print("saving fail")
        ret_dict = self.ground_eval(annotations, preds)

        return ret_dict


if __name__ == '__main__':

    result_file = torch.load(
        '/mnt/petrelfs/linjingli/tmp/data/big_tmp/result_og_es.pt')

    annotations = result_file['gt_list']
    preds = result_file['pred_list']

    test_eval = GroundingMetricMod()
    test_eval.ground_eval(annotations, preds)
