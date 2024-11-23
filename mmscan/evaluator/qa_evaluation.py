import torch

from mmscan.evaluator.metrics.lang_metric import (coco_evaluate, em_evaluation,
                                                  sbert_evaluator,
                                                  simcse_evaluator)
from mmscan.utils.lang_utils import special_token_filter


class QA_Evaluator:
    """Tradition metrics for QA and Caption evaluation , consists the
    implements of.

       [EM, BLEU, METEOR, ROUGE, CIDEr, SPICE, SIMCSE, SBERT]
       SIMCSE, SBERT is speacial metrics and needed GPU.

    Attributes:
        save_buffer(list[dict]): Save the buffer of Inputs.
        records(list[dict]): Metric results for each sample.
        metric_record(dict): Metric results for each category.
            (average of all samples with the same category)
    Args:
        model_config(dict): The model config for special metric evaluation.
            Defaults to {}.
        max_length(int): The maximum length of the input.
            Defaults to 1024.
        verbose(bool): Whether to print the evaluation results.
            Defaults to True.
    """

    def __init__(self, model_config={}, max_length=256, verbose=True) -> None:
        self.eval_bs = 500
        self.verbose = verbose
        self.max_length = max_length
        self.special_metric = []
        if 'simcse' in model_config and torch.cuda.is_available():
            self.special_metric.append('simcse')
            self.simcse_evaluator = simcse_evaluator(model_config['simcse'],
                                                     eval_bs=self.eval_bs)
        if 'sbert' in model_config and torch.cuda.is_available():
            self.special_metric.append('sbert')
            self.sbert_evaluator = sbert_evaluator(model_config['sbert'],
                                                   eval_bs=self.eval_bs)

        self.eval_metric = [
            'EM',
            'refined_EM',
            'Bleu_1',
            'Bleu_2',
            'Bleu_3',
            'Bleu_4',
            'METEOR',
            'ROUGE_L',
            'CIDEr',
            'SPICE',
        ] + self.special_metric

        self.reset()

    def reset(self):
        """Reset the evaluator, clear the buffer and records."""
        self.metric_record = {}
        self.save_results = {}
        self.save_buffer = []
        self.records = []

    def update(self, batch_input):
        """Update a batch of results to the buffer, and then filtering and
        truncating. each item is expected to be a dict with keys.

        ["index", "ID","question","pred","gt"]

        1. pred is a list with one one element.
        2. gt is a list with >=1 elements.
        3. "ID" should be unique.

        Args:
            batch_input (list[dict]):
                Batch of the raw original input.
        Returns:
            Dict: {"EM":EM metric for this batch,
                "refined_EM":Refined EM metric for this batch}
        """

        self.__check_format__(batch_input)

        for _input in batch_input:
            _input['pred'] = [
                special_token_filter(
                    _input['pred'][0],
                    clean=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            ]
            _input['gt'] = [
                special_token_filter(i,
                                     clean=True,
                                     truncation=True,
                                     max_length=self.max_length)
                for i in _input['gt']
            ]

        self.save_buffer.extend(batch_input)

        EM_, refine_EM_ = em_evaluation(batch_input)
        return {
            'EM': sum(EM_) / len(EM_),
            'refined_EM': sum(refine_EM_) / len(refine_EM_),
        }

    def start_evaluation(self):
        """Start the evaluation process.

        Returns:
            dict: The results of the evaluation.
        """

        # (1) exact match evaluation
        EM_, refine_EM_ = em_evaluation(self.save_buffer)

        # (2) coco metric evaluation
        coco_scores, coco_scores_list = coco_evaluate(self.save_buffer)

        # (3) special metric evaluation, forward one time each batch
        if 'simcse' in self.special_metric:
            all_simcse_similarity = self.simcse_evaluator.evaluation(
                self.save_buffer)
        if 'sbert' in self.special_metric:
            all_sbert_similarity = self.sbert_evaluator.evaluation(
                self.save_buffer)

        # (1) store for every sample
        store_dict = {'EM': EM_, 'refined_EM': refine_EM_}

        for metric_ in coco_scores:
            if metric_ == 'SPICE':
                store_dict[metric_] = [
                    item['All'] for item in coco_scores_list['SPICE']
                ]
            else:
                store_dict[metric_] = coco_scores_list[metric_]
        if 'simcse' in self.special_metric:
            store_dict['simcse'] = all_simcse_similarity
        if 'sbert' in self.special_metric:
            store_dict['sbert'] = all_sbert_similarity

        for _index, item_dict in enumerate(self.save_buffer):
            for metric in store_dict:
                item_dict[metric] = store_dict[metric][_index]

            self.records.append(item_dict)

        # (2) return the final mean metric

        eval_dict = {}
        for metric in self.eval_metric:
            if metric not in coco_scores.keys():
                eval_dict.update({
                    metric:
                    sum(store_dict[metric]) / len(store_dict[metric])
                })
            else:
                eval_dict[metric] = coco_scores[metric]
        self.metric_record = eval_dict

        if self.verbose:
            print(eval_dict)

        return eval_dict

    def __check_format__(self, raw_input):
        """Check if the input conform with mmscan evaluation format.

        Every item with the keys ["index", "ID","question","pred","gt"],
            'pred' is a list with one one element, 'gt' is a list
            with >=1 elements. "ID" should be unique.
        Args:
            raw_input (list[dict]): The input to be checked.
        """
        assert isinstance(
            raw_input,
            list), 'The input of QA evaluator should be a list of dict. '

        for _index in range(len(raw_input)):
            if 'index' not in raw_input[_index]:
                raw_input[_index]['index'] = len(self.save_buffer) + _index

            assert 'ID' in raw_input[_index]
            assert ('pred' in raw_input[_index]
                    and isinstance(raw_input[_index]['pred'], list)
                    and len(raw_input[_index]['pred']) == 1)
            assert ('gt' in raw_input[_index]
                    and isinstance(raw_input[_index]['gt'], list)
                    and len(raw_input[_index]['gt']) >= 1)
            assert 'question' in raw_input[_index]
