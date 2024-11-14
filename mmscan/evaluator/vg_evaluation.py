import numpy as np
import torch
from terminaltables import AsciiTable
from tqdm import tqdm

from mmscan.evaluator.metrics.box_metric import (compute_for_subset,
                                                 get_average_precision,
                                                 get_multi_topk_scores)
from mmscan.utils.box_utils import euler_iou3d_bbox, euler_to_matrix_np


class VG_Evaluator:
    """Evaluator for MMScan Visual Grounding benchmark.

    Attributes:
        eval_metric: All the evaluation metric, includes
            "AP","AP_C","AR","gTop-k"
        save_buffer(list[dict]): Save the buffer of Inputs

        records(list[dict]): Metric results for each sample

        category_records(dict): Metric results for each category
            (average of all samples with the same category)
    Args:
        verbose(bool): Whether to print the evaluation results.
            Defaults to True.
    """

    def __init__(self, verbose=True) -> None:
        self.verbose = verbose
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

    def reset(self):
        self.save_buffer = []
        self.records = []
        self.category_records = {}

    def update(self, raw_batch_input):
        """Update a batch of results to the buffer.

        Args:
            raw_batch_input (list[dict]):
            a batch of the raw original input
        """
        self.__check_format__(raw_batch_input)
        self.save_buffer.extend(raw_batch_input)

    def start_evaluation(self):
        """This function is used to start the evaluation process.

        It will iterate over the saved buffer and evaluate each item.
        """

        category_collect = {}

        for data_item in tqdm(self.save_buffer):
            metric_for_single = {}

            # (1) len(gt)==0 : skip
            if len(data_item['gt_bboxes']['center']) == 0:
                continue

            # (2) len(pred)==0 : model's wrong
            if len(data_item['pred_bboxes']['center']) == 0:
                for iou_thr in self.iou_thresholds:
                    metric_for_single[f'AP@{iou_thr}'] = 0
                    for topk in [1, 3, 5, 10]:
                        metric_for_single[f'gTop-{topk}@{iou_thr}'] = 0

                data_item['num_gts'] = len(data_item['gt_bboxes']['center'])
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
                    AP_C, AR_C = compute_for_subset(category_collect[category],
                                                    iou_thr)
                    self.category_records[category][f'AP_C@{iou_thr}'] = AP_C
                    self.category_records[category][f'AR_C@{iou_thr}'] = AR_C
                    self.category_records['overall'][f'AP_C@{iou_thr}'] += (
                        AP_C * category_collect[category]['cnt'] /
                        len(self.records))
                    self.category_records['overall'][f'AR_C@{iou_thr}'] += (
                        AR_C * category_collect[category]['cnt'] /
                        len(self.records))

        return self.category_records

    def collect_result(self):
        """Collect the result from the evaluation process.

        Stores them based on some subclass.
        Returns:
             category_results(dict): Average results per category
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

    def print_result(self):
        """Showing the result table.

        Returns:
            table(str): the metric result table
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

        if self.verbose:
            print(table.table)

        return table.table

    def __category_mapping__(self, sub_class):
        """Mapping the subclass name to the category name.

        Args:
            sub_class (str): the subclass name in the original samples

        Returns:
            category (str): the category name.
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

    def __calculate_iou_array_(self, data_item):
        """Calculate some information needed for eavl.

        Args:
             data_item (dict): the subclass name in the original samples
        Returns:
             nd.array, int, int :
                the iou array sorted by the confidence,
                number of predictions, number of gts.
        """

        pred_bboxes = data_item['pred_bboxes']
        gt_bboxes = data_item['gt_bboxes']
        # Sort the bounding boxes based on their scores
        pred_scores = data_item['pred_scores']
        top_idxs = torch.argsort(pred_scores, descending=True)
        pred_scores = pred_scores[top_idxs]

        pred_center = pred_bboxes['center'][top_idxs]
        pred_size = pred_bboxes['size'][top_idxs]
        pred_rot = pred_bboxes['rot'][top_idxs]

        gt_center = gt_bboxes['center']
        gt_size = gt_bboxes['size']
        gt_rot = gt_bboxes['rot']

        # shape (num_pred , num_gt)
        iou_matrix = euler_iou3d_bbox(pred_center, pred_size, pred_rot,
                                      gt_center, gt_size, gt_rot)

        # (3) calculate the TP and NP,
        # preparing for the forward AP/topk calculation
        pred_scores = pred_scores.cpu().numpy()
        sorted_inds = np.argsort(-pred_scores)
        iou_array = np.array([iou_matrix[i] for i in sorted_inds])

        return iou_array, pred_scores

    def __check_format__(self, raw_input):
        """Check if the input conform with mmscan evaluation format.

        transform 9 DoF box to ('center'/'size'/'rot_matrix')
        """
        assert isinstance(
            raw_input,
            list), 'The input of MMScan evaluator should be a list of dict. '
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
                    continue
                center_list = []
                size_list = []
                angle_list = []
                rot_list = []
                for box_index in range(len(raw_input[_index][mode])):
                    _9_dof_box_ = (
                        raw_input[_index][mode][box_index].cpu().numpy())
                    if len(_9_dof_box_.shape) > 1:
                        _9_dof_box_ = _9_dof_box_[0]
                    center_list.append(_9_dof_box_[:3])
                    size_list.append(_9_dof_box_[3:6])
                    angle_list.append(_9_dof_box_[6:])
                if len(angle_list) > 0:
                    rot_list = euler_to_matrix_np(np.array(angle_list))
                raw_input[_index][mode] = {
                    'center':
                    torch.tensor(np.array(center_list)).to(torch.float32),
                    'size':
                    torch.tensor(np.array(size_list)).to(torch.float32),
                    'rot': torch.tensor(np.array(rot_list)).to(torch.float32),
                }
