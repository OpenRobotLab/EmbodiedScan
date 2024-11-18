import json
from pathlib import Path

import numpy as np
from accelerate.logging import get_logger
from data.eai import (_CLIPORT_ACTION_SPACE_U, _CLIPORT_ACTION_SPACE_V,
                      _CLIPORT_ACTION_SPACE_ZROT, _DUMMY_CLIPORT_ACTION,
                      CLIPORT_ACTION_SPACE_DETOKENIZE)
from evaluator.build import EVALUATOR_REGISTRY

logger = get_logger(__name__)


@EVALUATOR_REGISTRY.register()
class ObjNavEvaluator():

    def __init__(self, cfg, task_name):
        self.task_name = task_name
        self.best_result = -np.inf
        self.save_dir = Path(cfg.exp_dir) / 'eval_results' / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self.eval_dict = {'target_metric': [], 'accuracy': []}
        self.total_count = 0
        self.save_results = []

    def batch_metrics(self, data_dict):
        metrics = {}
        preds = data_dict['output_txt']
        gts = data_dict['output_gt']

        correct = 0
        for pred, gt in zip(preds, gts):
            if pred == gt:
                correct += 1

        batch_size = len(gts)
        metrics['total_count'] = batch_size
        metrics['accuracy'] = correct / batch_size
        metrics['target_metric'] = metrics['accuracy']
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size

        for i in range(batch_size):
            self.save_results.append({
                # vision
                'source':
                data_dict['source'][i],
                'scene_id':
                data_dict['scene_id'][i],
                # language
                'instruction':
                data_dict['prompt_after_obj'][i],
                'response_gt':
                data_dict['output_gt'][i],
                'response_pred':
                data_dict['output_txt'][i],
            })

        for key in self.eval_dict.keys():
            self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False

        if (is_best or split == 'test') and is_main_process:
            with open(str(self.save_dir / 'results.json'), 'w') as f:
                json.dump(self.save_results, f, indent=2)

        return is_best, self.eval_dict


@EVALUATOR_REGISTRY.register()
class CLIPortEvaluator(ObjNavEvaluator):

    def reset(self):
        self.eval_dict = {
            'target_metric': [],
            'accuracy': [],
            'action_error_pose0': [],
            'action_error_pose1': [],
        }
        self.total_count = 0
        self.save_results = []

    def batch_metrics(self, data_dict):
        metrics = super().batch_metrics(data_dict)

        # add action errors (for xy coordinates) to metrics
        metrics['action_error_pose0'] = []
        metrics['action_error_pose1'] = []
        for pred, gt in zip(data_dict['output_txt'], data_dict['output_gt']):
            action_pred = CLIPortEvaluator.parse_action_cliport(pred)
            action_gt = CLIPortEvaluator.parse_action_cliport(gt)
            for pose_name in ['pose0', 'pose1']:
                xy_pred = action_pred[pose_name][0][:2]
                xy_gt = action_gt[pose_name][0][:2]
                error = np.linalg.norm(xy_pred - xy_gt)
                metrics[f'action_error_{pose_name}'].append(error)
        metrics['action_error_pose0'] = np.mean(metrics['action_error_pose0'])
        metrics['action_error_pose1'] = np.mean(metrics['action_error_pose1'])

        return metrics

    @staticmethod
    def parse_action_cliport(text, obs=None):
        vocab = list(_CLIPORT_ACTION_SPACE_U.values()) + list(
            _CLIPORT_ACTION_SPACE_V.values()) + list(
                _CLIPORT_ACTION_SPACE_ZROT.values())
        tokens = CLIPortEvaluator.tokenize(text, '', vocab)
        if len(tokens) != 6:
            logger.info(f'Cannot parse action: {text}')
            return _DUMMY_CLIPORT_ACTION
        try:
            pose0 = CLIPORT_ACTION_SPACE_DETOKENIZE(tokens[:3], obs)
            pose1 = CLIPORT_ACTION_SPACE_DETOKENIZE(tokens[3:], obs)
            return {
                'pose0': pose0,
                'pose1': pose1,
            }
        except Exception as e:
            logger.info(f'{e}, cannot parse action: {text}')
            return _DUMMY_CLIPORT_ACTION

    @staticmethod
    def tokenize(text, prefix, vocab):
        # Find the starting index of the prefix in the text
        start_index = text.find(prefix)

        # If the prefix is not found, return an empty list
        if start_index == -1:
            return []

        # Get the text after the prefix
        after_prefix = text[start_index + len(prefix):]

        # Create a list to hold the token ids
        tokens = []

        # Iterate over each character in after_prefix
        i = 0
        while i < len(after_prefix):
            # Try to find the longest token in vocab that matches the current position
            match = None
            max_length = 0
            for token in vocab:
                token_length = len(token)
                if after_prefix[
                        i:i +
                        token_length] == token and token_length > max_length:
                    match = token
                    max_length = token_length
            if match is not None:
                tokens.append(match)
                i += max_length
            else:
                i += 1

        return tokens
