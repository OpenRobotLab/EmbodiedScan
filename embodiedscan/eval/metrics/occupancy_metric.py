# Copyright (c) OpenRobotLab. All rights reserved.
import logging
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from embodiedscan.registry import METRICS


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (list[float]): List of iou threshold when calculate the
            metric. Defaults to  [0.25, 0.5].
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 batchwise_anns: bool = False,
                 **kwargs):
        super(OccupancyMetric, self).__init__(prefix=prefix,
                                              collect_device=collect_device)
        self.batchwise_anns = batchwise_anns

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_occ = data_sample['pred_occupancy']
            gt_4 = data_sample['gt_occupancy']
            gt_occ = torch.zeros_like(pred_occ)
            gt_occ[gt_4[:, 0], gt_4[:, 1], gt_4[:, 2]] = gt_4[:, 3]
            if 'gt_occupancy_masks' in data_sample:
                gt_occ_mask = data_sample['gt_occupancy_masks']
                gt_occ[~gt_occ_mask] = 255
            self.results.append((gt_occ, pred_occ))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        num_class = len(self.dataset_meta['classes']) + 1
        score = np.zeros((num_class, 3))

        for gt_occ, sinlge_pred_results in results:
            mask = (gt_occ != 255)
            for j in range(num_class):
                if j == 0:  # class 0 (empty) for geometry IoU
                    score[j][0] += ((gt_occ[mask] != 0) *
                                    (sinlge_pred_results[mask] != 0)).sum()
                    score[j][1] += (gt_occ[mask] != 0).sum()
                    score[j][2] += (sinlge_pred_results[mask] != 0).sum()
                else:
                    score[j][0] += ((gt_occ[mask] == j) *
                                    (sinlge_pred_results[mask] == j)).sum()
                    score[j][1] += (gt_occ[mask] == j).sum()
                    score[j][2] += (sinlge_pred_results[mask] == j).sum()

        ret_dict = dict()
        table_data = [['classes', 'IoU']]
        res = []
        for i in range(num_class):
            name = 'empty'
            if i > 0:
                name = self.dataset_meta['classes'][i - 1]

            tp = score[i, 0]
            p = score[i, 1]
            g = score[i, 2]
            union = p + g - tp
            # do not save the accuracy result if nan
            if np.isnan(tp / union):
                continue
            ret_dict[name] = tp / union
            res.append(tp / union)
            table_data.append([name, f'{ret_dict[name]:.5f}'])
        table_data.append(['mean', f'{sum(res)/len(res):.5f}'])

        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        return ret_dict

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.batchwise_anns:
            # the actual dataset length/size is the len(self.results)
            if self.collect_device == 'cpu':
                results = collect_results(self.results,
                                          len(self.results),
                                          self.collect_device,
                                          tmpdir=self.collect_dir)
            else:
                results = collect_results(self.results, len(self.results),
                                          self.collect_device)
        else:
            if self.collect_device == 'cpu':
                results = collect_results(self.results,
                                          size,
                                          self.collect_device,
                                          tmpdir=self.collect_dir)
            else:
                results = collect_results(self.results, size,
                                          self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]
