from typing import List, Tuple

import numpy as np
import torch
from terminaltables import AsciiTable
from tqdm import tqdm

from mmscan.evaluator.metrics.box_metric import (get_average_precision,
                                                 get_multi_topk_scores,
                                                 subset_get_average_precision)
from mmscan.utils.box_utils import index_box, to_9dof_box


class VG_Evaluator:
    """Evaluator for MMScan Visual Grounding benchmark. The evaluation metric
    includes "AP","AP_C","AR","gTop-k".

    Attributes:
        save_buffer(list[dict]): Save the buffer of Inputs.

        records(list[dict]): Metric results for each sample

        category_records(dict): Metric results for each category
            (average of all samples with the same category)
    Args:
        show_results(bool): Whether to print the evaluation results.
            Defaults to True.
    """

    def __init__(self, show_results: bool = True) -> None:

        self.show_results = show_results
        self.eval_metric_type = ['AP', 'AR']
        self.top_k_visible = [1, 3, 5]
        self.call_for_category_mode = True

        for top_k in [1, 3, 5, 10]:
            self.eval_metric_type.append(f'gTop-{top_k}')

        self.iou_thresholds = [0.25, 0.50]
        self.eval_metric = []
        for iou_thr in self.iou_thresholds:
            for eval_type in self.eval_metric_type:
                self.eval_metric.append(eval_type + '@' + str(iou_thr))

        self.reset()

    def reset(self) -> None:
        """Reset the evaluator, clear the buffer and records."""
        self.save_buffer = []
        self.records = []
        self.category_records = {}

    def update(self, raw_batch_input: List[dict]) -> None:
        """Update a batch of results to the buffer.

        Args:
            raw_batch_input (list[dict]):
                Batch of the raw original input.
        """
        self.__check_format__(raw_batch_input)
        self.save_buffer.extend(raw_batch_input)

    def start_evaluation(self) -> dict:
        """This function is used to start the evaluation process.

        It will iterate over the saved buffer and evaluate each item.
        Returns:
             category_records(dict): Metric results per category.
        """

        category_collect = {}

        for data_item in tqdm(self.save_buffer):
            metric_for_single = {}

            # (1) len(gt)==0 : skip
            if self.__is_zero__(data_item['gt_bboxes']):
                continue

            # (2) len(pred)==0 : model's wrong
            if self.__is_zero__(data_item['pred_bboxes']):
                for iou_thr in self.iou_thresholds:
                    metric_for_single[f'AP@{iou_thr}'] = 0
                    for topk in [1, 3, 5, 10]:
                        metric_for_single[f'gTop-{topk}@{iou_thr}'] = 0

                data_item['num_gts'] = len(data_item['gt_bboxes'])
                data_item.update(metric_for_single)
                self.records.append(data_item)
                continue

            iou_array, pred_score = self.__calculate_iou_array_(data_item)
            if self.call_for_category_mode:
                category = self.__category_mapping__(data_item['subclass'])
                if category not in category_collect.keys():
                    category_collect[category] = {
                        'ious': [],
                        'scores': [],
                        'sample_indices': [],
                        'cnt': 0,
                    }

                category_collect[category]['ious'].extend(iou_array)
                category_collect[category]['scores'].extend(pred_score)
                category_collect[category]['sample_indices'].extend(
                    [data_item['index']] * len(iou_array))
                category_collect[category]['cnt'] += 1

            for iou_thr in self.iou_thresholds:
                # AP/AR metric
                AP, AR = get_average_precision(iou_array, iou_thr)
                metric_for_single[f'AP@{iou_thr}'] = AP
                metric_for_single[f'AR@{iou_thr}'] = AR

                # topk metric
                metric_for_single.update(
                    get_multi_topk_scores(iou_array, iou_thr))

            data_item['num_gts'] = iou_array.shape[1]
            data_item.update(metric_for_single)
            self.records.append(data_item)

        self.collect_result()

        if self.call_for_category_mode:
            for iou_thr in self.iou_thresholds:
                self.category_records['overall'][f'AP_C@{iou_thr}'] = 0
                self.category_records['overall'][f'AR_C@{iou_thr}'] = 0

                for category in category_collect:
                    AP_C, AR_C = subset_get_average_precision(
                        category_collect[category], iou_thr)
                    self.category_records[category][f'AP_C@{iou_thr}'] = AP_C
                    self.category_records[category][f'AR_C@{iou_thr}'] = AR_C
                    self.category_records['overall'][f'AP_C@{iou_thr}'] += (
                        AP_C * category_collect[category]['cnt'] /
                        len(self.records))
                    self.category_records['overall'][f'AR_C@{iou_thr}'] += (
                        AR_C * category_collect[category]['cnt'] /
                        len(self.records))

        return self.category_records

    def collect_result(self) -> dict:
        """Collect the result from the evaluation process.

        Stores them based on their subclass.
        Returns:
             category_results(dict): Average results per category.
        """
        category_results = {}
        category_results['overall'] = {}

        for metric_name in self.eval_metric:
            category_results['overall'][metric_name] = []
            category_results['overall']['num_gts'] = 0

        for data_item in self.records:
            category = self.__category_mapping__(data_item['subclass'])

            if category not in category_results:
                category_results[category] = {}
                for metric_name in self.eval_metric:
                    category_results[category][metric_name] = []
                    category_results[category]['num_gts'] = 0

            for metric_name in self.eval_metric:
                for metric_name in self.eval_metric:
                    category_results[category][metric_name].append(
                        data_item[metric_name])
                    category_results['overall'][metric_name].append(
                        data_item[metric_name])

            category_results['overall']['num_gts'] += data_item['num_gts']
            category_results[category]['num_gts'] += data_item['num_gts']
        for category in category_results:
            for metric_name in self.eval_metric:
                category_results[category][metric_name] = np.mean(
                    category_results[category][metric_name])

        self.category_records = category_results

        return category_results

    def print_result(self) -> str:
        """Showing the result table.

        Returns:
            table(str): The metric result table.
        """
        assert len(self.category_records) > 0, 'No result yet.'
        self.category_records = {
            key: self.category_records[key]
            for key in sorted(self.category_records.keys(), reverse=True)
        }

        header = ['Type']
        header.extend(self.category_records.keys())
        table_columns = [[] for _ in range(len(header))]

        # some metrics
        for iou_thr in self.iou_thresholds:
            show_in_table = (['AP', 'AR'] +
                             [f'gTop-{k}' for k in self.top_k_visible]
                             if not self.call_for_category_mode else
                             ['AP', 'AR', 'AP_C', 'AR_C'] +
                             [f'gTop-{k}' for k in self.top_k_visible])

            for metric_type in show_in_table:
                table_columns[0].append(metric_type + ' ' + str(iou_thr))

            for i, category in enumerate(self.category_records.keys()):
                ap = self.category_records[category][f'AP@{iou_thr}']
                ar = self.category_records[category][f'AR@{iou_thr}']
                table_columns[i + 1].append(f'{float(ap):.4f}')
                table_columns[i + 1].append(f'{float(ar):.4f}')

                ap = self.category_records[category][f'AP_C@{iou_thr}']
                ar = self.category_records[category][f'AR_C@{iou_thr}']
                table_columns[i + 1].append(f'{float(ap):.4f}')
                table_columns[i + 1].append(f'{float(ar):.4f}')
                for k in self.top_k_visible:
                    top_k = self.category_records[category][
                        f'gTop-{k}@{iou_thr}']
                    table_columns[i + 1].append(f'{float(top_k):.4f}')

        # Number of gts
        table_columns[0].append('Num GT')
        for i, category in enumerate(self.category_records.keys()):
            table_columns[i + 1].append(
                f'{int(self.category_records[category]["num_gts"])}')

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table_data = [list(row) for row in zip(*table_data)]
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True

        if self.show_results:
            print(table.table)

        return table.table

    def __category_mapping__(self, sub_class: str) -> str:
        """Mapping the subclass name to the category name.

        Args:
            sub_class (str): The subclass name in the original samples.

        Returns:
            category (str): The category name.
        """
        sub_class = sub_class.lower()
        sub_class = sub_class.replace('single', 'sngl')
        sub_class = sub_class.replace('inter', 'int')
        sub_class = sub_class.replace('unique', 'uniq')
        sub_class = sub_class.replace('common', 'cmn')
        sub_class = sub_class.replace('attribute', 'attr')
        if 'sngl' in sub_class and ('attr' in sub_class or 'eq' in sub_class):
            sub_class = 'vg_sngl_attr'
        return sub_class

    def __calculate_iou_array_(
            self, data_item: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate some information needed for eavl.

        Args:
             data_item (dict): The subclass name in the original samples.
        Returns:
             np.ndarray, np.ndarray :
                The iou array sorted by the confidence and the
                confidence scores.
        """

        pred_bboxes = data_item['pred_bboxes']
        gt_bboxes = data_item['gt_bboxes']
        # Sort the bounding boxes based on their scores
        pred_scores = data_item['pred_scores']
        top_idxs = torch.argsort(pred_scores, descending=True)
        pred_scores = pred_scores[top_idxs]

        pred_bboxes = to_9dof_box(index_box(pred_bboxes, top_idxs))
        gt_bboxes = to_9dof_box(gt_bboxes)

        iou_matrix = pred_bboxes.overlaps(pred_bboxes,
                                          gt_bboxes)  # (num_query, num_gt)
        # (3) calculate the TP and NP,
        # preparing for the forward AP/topk calculation
        pred_scores = pred_scores.cpu().numpy()
        iou_array = iou_matrix.cpu().numpy()

        return iou_array, pred_scores

    def __is_zero__(self, box):
        if isinstance(box, (list, tuple)):
            return (len(box[0]) == 0)
        return (len(box) == 0)

    def __check_format__(self, raw_input: List[dict]) -> None:
        """Check if the input conform with mmscan evaluation format. Transform
        the input box format.

        Args:
            raw_input (list[dict]): The input of VG evaluator.
        """
        assert isinstance(
            raw_input,
            list), 'The input of VG evaluator should be a list of dict. '
        raw_input = raw_input

        for _index in tqdm(range(len(raw_input))):
            if 'index' not in raw_input[_index]:
                raw_input[_index]['index'] = len(self.save_buffer) + _index

            if 'subclass' not in raw_input[_index]:
                raw_input[_index]['subclass'] = 'non-class'

            assert 'gt_bboxes' in raw_input[_index]
            assert 'pred_bboxes' in raw_input[_index]
            assert 'pred_scores' in raw_input[_index]

            for mode in ['pred_bboxes', 'gt_bboxes']:
                if (isinstance(raw_input[_index][mode], dict)
                        and 'center' in raw_input[_index][mode]):
                    raw_input[_index][mode] = [
                        torch.tensor(raw_input[_index][mode]['center']),
                        torch.tensor(raw_input[_index][mode]['size']).to(
                            torch.float32),
                        torch.tensor(raw_input[_index][mode]['rot']).to(
                            torch.float32)
                    ]
