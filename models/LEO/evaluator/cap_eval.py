import json
from pathlib import Path

import numpy as np
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.ngram_metrics.bleu.bleu import Bleu
from evaluator.ngram_metrics.cider.cider import Cider
from evaluator.ngram_metrics.rouge.rouge import Rouge
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim


@EVALUATOR_REGISTRY.register()
class CaptionEvaluator():

    def __init__(self, cfg, task_name):
        self.task_name = task_name

        self.cider_scorer = Cider()
        self.bleu_scorer = Bleu()
        self.rouge_scorer = Rouge()

        self.best_result = -np.inf

        self.save_dir = Path(cfg.exp_dir) / 'eval_results' / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_path = cfg.data.scan2cap.corpus if self.task_name.lower(
        ) == 'scan2cap' else None
        self.sentence_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2')

        self.reset()

    def reset(self):
        self.eval_dict = {
            'target_metric': [],
            'sentence_sim': [],
            'cider': 0,
            'bleu': 0,
            'rouge': 0,
        }
        self.total_count = 0
        self.save_results = []
        self.init_corpus()

    def init_corpus(self):
        if self.task_name.lower() == 'scan2cap':
            with open(self.corpus_path, 'r') as f:
                self.gt_sentence_mp = json.load(f)
            self.pred_sentence_mp = {}
        else:
            # init with list, finally convert to dict
            self.gt_sentence_mp = []
            self.pred_sentence_mp = []

    def batch_metrics(self, data_dict):
        metrics = {}
        output_gt = data_dict['output_gt']
        output_pred = data_dict['output_txt']
        batch_size = len(output_gt)

        # consider IoU-based caption metrics
        if 'iou_flag' in data_dict:
            iou_flags = data_dict['iou_flag']
        else:
            iou_flags = [True] * batch_size

        if self.task_name.lower() == 'scan2cap':
            for i in range(batch_size):
                corpus_key = data_dict['corpus_key'][i]
                if iou_flags[i]:
                    self.pred_sentence_mp[corpus_key] = [
                        ('sos ' + output_pred[i] + ' eos').replace(
                            '. ', ' . ')
                    ]
                else:
                    output_pred[i] = ''
                    self.pred_sentence_mp[corpus_key] = ['sos eos']
        else:
            for i in range(batch_size):
                if iou_flags[i]:
                    self.pred_sentence_mp.append([output_pred[i]])
                else:
                    output_pred[i] = ''
                    self.pred_sentence_mp.append([''])
                self.gt_sentence_mp.append([output_gt[i]])

        # compute sentence similarity
        embed_pred = self.sentence_model.encode(output_pred,
                                                convert_to_tensor=True)
        embed_gt = self.sentence_model.encode(output_gt,
                                              convert_to_tensor=True)
        sims = pytorch_cos_sim(embed_pred, embed_gt).diag()

        metrics['total_count'] = batch_size
        metrics['sentence_sim'] = sims.mean().item()
        metrics['target_metric'] = metrics['sentence_sim']
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size

        for i in range(batch_size):
            save_dict = {
                # vision
                'source': data_dict['source'][i],
                'scene_id': data_dict['scene_id'][i],
                'anchor': data_dict['anchor_locs'][i].tolist(),
                # language
                'instruction': data_dict['prompt_after_obj'][i],
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
            }
            if 'iou_flag' in data_dict:
                save_dict['iou_flag'] = data_dict['iou_flag'][i].item()
            self.save_results.append(save_dict)

        for key in self.eval_dict.keys():
            if key not in ['cider', 'bleu', 'rouge']:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # ngram metrics
        if self.task_name.lower() == 'scan2cap':
            # align gt_sentence_mp to pred_sentence_mp for partial evaluation
            self.gt_sentence_mp = {
                k: self.gt_sentence_mp[k]
                for k in self.pred_sentence_mp.keys()
            }
        else:
            self.gt_sentence_mp = {
                k: v
                for k, v in enumerate(self.gt_sentence_mp)
            }
            self.pred_sentence_mp = {
                k: v
                for k, v in enumerate(self.pred_sentence_mp)
            }

        self.eval_dict['cider'] = self.cider_scorer.compute_score(
            self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(
            self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]

        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(
            self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k not in ['cider', 'bleu', 'rouge']:
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
