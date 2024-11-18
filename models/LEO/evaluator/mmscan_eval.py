import json
from pathlib import Path

from mmscan import QA_Evaluator

model_config = {'simcse': '', 'sbert': ''}

from evaluator.build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class MMScanEvaluator():

    def __init__(self, cfg, task_name):
        self.evaluator = QA_Evaluator(model_config)
        self.task_name = task_name
        self.target_metric = 'refined_EM'
        self.best_result = 0.0
        self.save_dir = Path('./logs') / 'eval_results' / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def to_mmscan_form(self, raw_input):

        _input = {}
        _input['ID'] = raw_input['question_id']
        if not isinstance(raw_input['output_txt'], list):
            _input['pred'] = [raw_input['output_txt']]
        else:
            _input['pred'] = raw_input['output_txt']
        if not isinstance(raw_input['output_gt'], list):

            _input['gt'] = [raw_input['output_gt']]
        else:
            _input['gt'] = raw_input['output_gt']
        _input['question'] = raw_input['prompt_after_obj']

        return _input

    def reset(self):
        self.evaluator.reset()

    def update(self, raw_input_dict):
        # update buffer
        raw_input_dict = flatten_items(raw_input_dict)
        batch_input = []
        for _input in raw_input_dict:
            batch_input.append(self.to_mmscan_form(_input))

        self.evaluator.update(batch_input)

    @property
    def save_results(self):
        return self.evaluator.save_buffer

    def record(self, split, is_main_process):
        # record after a whole epoch
        self.evaluator.start_evaluation()

        results = self.evaluator.metric_record

        score = results[self.target_metric]
        results['target_metric'] = score

        if score > self.best_result:
            is_best = True
            self.best_result = score
        else:
            is_best = False

        if (is_best or split == 'test') and is_main_process:
            with open(str(self.save_dir / 'results.json'), 'w') as f:
                json.dump(self.save_results, f, indent=2)

        return is_best, results


def flatten_items(raw_input_dict):
    batch_input = []
    for _index in range(len(raw_input_dict['question_id'])):
        _input = {}
        for key_name in raw_input_dict:
            _input[key_name] = raw_input_dict[key_name][_index]

        batch_input.append(_input)
    return batch_input
