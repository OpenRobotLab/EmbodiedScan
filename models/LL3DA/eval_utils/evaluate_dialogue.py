import datetime
import importlib
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch
import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.meteor.meteor as capmeteor
import utils.capeval.rouge.rouge as caprouge
from utils.ap_calculator import APCalculator
from utils.box_util import box3d_iou_batch_tensor
from utils.dist import (all_gather_dict, all_reduce_average, barrier, get_rank,
                        init_distributed, is_distributed, is_primary)
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions


def score_captions(corpus: dict, candidates: dict):

    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    score_per_caption = {
        'bleu-1': [float(s) for s in bleu[1][0]],
        'bleu-2': [float(s) for s in bleu[1][1]],
        'bleu-3': [float(s) for s in bleu[1][2]],
        'bleu-4': [float(s) for s in bleu[1][3]],
        'cider': [float(s) for s in cider[1]],
        'rouge': [float(s) for s in rouge[1]],
        'meteor': [float(s) for s in meteor[1]],
    }

    message = '\n'.join([
        '[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])),
        '[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])),
        '[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])),
        '[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])),
        '[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            cider[0], max(cider[1]), min(cider[1])),
        '[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            rouge[0], max(rouge[1]), min(rouge[1])),
        '[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}'.format(
            meteor[0], max(meteor[1]), min(meteor[1]))
    ])

    eval_metric = {
        'BLEU-4': bleu[0][3],
        'CiDEr': cider[0],
        'Rouge': rouge[0],
        'METEOR': meteor[0],
    }
    return score_per_caption, message, eval_metric


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):

    # prepare ground truth caption labels
    print('preparing corpus...')

    annotations = dataset_loader.dataset.annotations
    task_name = dataset_loader.dataset.task_name
    corpus = {
        '-'.join((anno['scene_id'], anno['question'])): anno['answers'] \
            for anno in annotations
    }
    candidates = {}
    ### initialize and prepare for evaluation
    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)

    model.eval()
    barrier()

    epoch_str = f'[{curr_epoch}/{args.max_epoch}]' if curr_epoch > 0 else ''

    for curr_iter, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
            'qformer_input_ids': batch_data_label['qformer_input_ids'],
            'qformer_attention_mask':
            batch_data_label['qformer_attention_mask'],
            'instruction': batch_data_label['instruction'],
            'instruction_mask': batch_data_label['instruction_mask'],
        }
        outputs = model(model_input, is_eval=True, task_name='chat')

        outputs = dict(output_ids=outputs['output_ids'], )

        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)

        output_ids = outputs['output_ids']  # batch x max_length
        answers = tokenizer.batch_decode(output_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

        sample_index = batch_data_label['scan_idx'].cpu().tolist()

        for idx in range(output_ids.shape[0]):
            anno = annotations[sample_index[idx]]
            key = '-'.join((anno['scene_id'], anno['question']))
            answer = answers[idx]
            answer = ' '.join(filter(lambda w: w, answer.split(' ')))
            candidates[key] = [answer]

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logout(f'Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; '
                   f'Evaluating on iter: {curr_train_iter}; '
                   f'Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB')
        barrier()

    # end of forward pass traversion
    score_per_caption, message, eval_metric = score_captions(
        OrderedDict([(key, corpus[key]) for key in candidates]), candidates)

    if is_primary():
        logout('\n----------------------Evaluation-----------------------\n')
        logout(message)

        with open(
                os.path.join(args.checkpoint_dir,
                             f'{task_name}_corpus_val.json'), 'w') as f:
            json.dump(corpus, f, indent=4)

        with open(
                os.path.join(args.checkpoint_dir,
                             f'{task_name}_pred_val.json'), 'w') as f:
            json.dump(candidates, f, indent=4)

        with open(
                os.path.join(args.checkpoint_dir,
                             f'{task_name}_pred_gt_val.json'), 'w') as f:
            pred_gt_val = {}
            for scene_object_id, scene_object_id_key in enumerate(candidates):
                pred_gt_val[scene_object_id_key] = {
                    'pred': candidates[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                    'score': {
                        'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                        'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                        'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                        'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                        'CiDEr': score_per_caption['cider'][scene_object_id],
                        'rouge': score_per_caption['rouge'][scene_object_id],
                        'meteor': score_per_caption['meteor'][scene_object_id]
                    }
                }
            json.dump(pred_gt_val, f, indent=4)

    return eval_metric
