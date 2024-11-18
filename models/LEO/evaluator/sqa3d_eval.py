import json

from data.data_utils import clean_answer
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.scanqa_eval import ScanQAEvaluator


@EVALUATOR_REGISTRY.register()
class SQA3DEvaluator(ScanQAEvaluator):

    def reset(self):
        self.eval_dict = {
            'target_metric': [],
            'em_overall': [],
            'em_refined_overall': [],
            'em_type0': [],
            'em_refined_type0': [],
            'em_type1': [],
            'em_refined_type1': [],
            'em_type2': [],
            'em_refined_type2': [],
            'em_type3': [],
            'em_refined_type3': [],
            'em_type4': [],
            'em_refined_type4': [],
            'em_type5': [],
            'em_refined_type5': [],
            'cider_overall': 0,
            'bleu_overall': 0,
            'rouge_overall': 0,
        }
        self.total_count = 0
        self.type_count = {
            0: 1e-10,
            1: 1e-10,
            2: 1e-10,
            3: 1e-10,
            4: 1e-10,
            5: 1e-10
        }
        self.save_results = []
        self.gt_sentence_mp = []
        self.pred_sentence_mp = []

    def batch_metrics(self, data_dict):
        metrics = {
            'type0_count': 1e-10,
            'type1_count': 1e-10,
            'type2_count': 1e-10,
            'type3_count': 1e-10,
            'type4_count': 1e-10,
            'type5_count': 1e-10,
        }

        em_overall = 0
        em_refined_overall = 0
        em_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        em_refined_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for answer_pred, answer_gts, sqa_type in zip(data_dict['output_txt'],
                                                     data_dict['output_gt'],
                                                     data_dict['sqa_type']):
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred,
                                                         gts=answer_gts)
            em_overall += em_flag
            em_refined_overall += em_refined_flag

            sqa_type = int(sqa_type)  # 0-dim tensor to int
            em_type[sqa_type] += em_flag
            em_refined_type[sqa_type] += em_refined_flag
            metrics[f'type{sqa_type}_count'] += 1

            self.pred_sentence_mp.append([answer_pred])
            self.gt_sentence_mp.append(answer_gts)

        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em_overall'] = em_overall / batch_size
        metrics['em_refined_overall'] = em_refined_overall / batch_size
        for key in em_type.keys():
            metrics[
                f'em_type{key}'] = em_type[key] / metrics[f'type{key}_count']
            metrics[f'em_refined_type{key}'] = em_refined_type[key] / metrics[
                f'type{key}_count']

        metrics['target_metric'] = metrics['em_refined_overall']
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size
        for key in metrics.keys():
            if 'type' in key and 'count' in key:
                # type{x}_count
                self.type_count[int(key[4])] += metrics[key]

        for i in range(batch_size):
            self.save_results.append({
                # vision
                'source':
                data_dict['source'][i],
                'scene_id':
                data_dict['scene_id'][i],
                'anchor':
                data_dict['anchor_locs'][i].tolist(),
                'anchor_ort':
                data_dict['anchor_orientation'][i].tolist(),
                # language
                'situation':
                data_dict['situation'][i],
                'instruction':
                data_dict['prompt_after_obj'][i],
                'response_gt':
                data_dict['output_gt'][i],
                'response_pred':
                data_dict['output_txt'][i],
            })

        # save eval dict
        for key in self.eval_dict.keys():
            if key in ['cider_overall', 'bleu_overall', 'rouge_overall']:
                continue
            if 'type' in key:
                self.eval_dict[key].append(metrics[key] *
                                           metrics[f'type{key[-1]}_count'])
            else:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # ngram metrics
        self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
        self.pred_sentence_mp = {
            k: v
            for k, v in enumerate(self.pred_sentence_mp)
        }

        self.eval_dict['cider_overall'] = self.cider_scorer.compute_score(
            self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu_overall'] = self.bleu_scorer.compute_score(
            self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]

        self.eval_dict['rouge_overall'] = self.rouge_scorer.compute_score(
            self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k in ['cider_overall', 'bleu_overall', 'rouge_overall']:
                continue
            if 'type' in k:
                self.eval_dict[k] = sum(v) / self.type_count[int(k[-1])]
            else:
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
