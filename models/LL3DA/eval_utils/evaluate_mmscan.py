import datetime
import importlib
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch
from utils.ap_calculator import APCalculator
from utils.box_util import box3d_iou_batch_tensor
from utils.dist import (all_gather_dict, all_reduce_average, barrier, get_rank,
                        init_distributed, is_distributed, is_primary)
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions

from mmscan import QA_Evaluator

model_config = {
    'simcse': '/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/pc',
    'sbert': '/mnt/petrelfs/linjingli/mmscan_modelzoo-main/evaluation/st'
}


def to_mmscan_form(raw_input):
    _input = {}
    _input['ID'] = raw_input['ID'].split('@')[0]
    _input['question'] = raw_input['ID'].split('@')[1]
    _input['pred'] = raw_input['answer_pred']
    _input['gt'] = raw_input['answer_gt']

    return _input


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

    evaluator = QA_Evaluator(model_config)

    annotations = dataset_loader.dataset.annotations



    corpus = {
    '@'.join((anno['ID'], anno['question'])): anno['answers'] if 'answers' in anno else anno['caption']  \
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
            try:
                batch_data_label[key] = batch_data_label[key].to(net_device)
            except:
                continue
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
        outputs = model(model_input, is_eval=True, task_name='qa')

        outputs = dict(output_ids=outputs['output_ids'], )

        outputs = all_gather_dict(outputs)

        batch_data_label = all_gather_dict(batch_data_label)

        output_ids = outputs['output_ids']  # batch x max_length
        answers = tokenizer.batch_decode(output_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

        sample_index = batch_data_label['scan_idx'].cpu().tolist()

        # rewrite the batch_result is ok
        batch_results = []
        for idx in range(output_ids.shape[0]):
            raw_input_dict = {}

            anno = annotations[sample_index[idx]]
            key = '@'.join((anno['ID'], anno['question']))

            # for the multi-gpu evaluation, we need to make sure that the same question is not evaluated multiple times
            # This is caused by the distributed sampler, last several samples may be duplicated
            if key in candidates:
                continue

            answer = answers[idx]
            answer = ' '.join(filter(lambda w: w, answer.split(' ')))
            candidates[key] = answer

            raw_input_dict['ID'] = key
            raw_input_dict['answer_pred'] = [answer]
            raw_input_dict['answer_gt'] = corpus[key]
            batch_results.append(to_mmscan_form(raw_input_dict))

        evaluator.update(batch_results)

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logout(f'Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; '
                   f'Evaluating on iter: {curr_train_iter}; '
                   f'Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB')
            if curr_iter % 200 == 0:
                with open(
                        os.path.join(args.checkpoint_dir,
                                     f'qa_pred_gt_val_{curr_iter}.json'),
                        'w') as f:
                    pred_gt_val = {}
                    for index_, scene_object_id_key in enumerate(candidates):
                        pred_gt_val[scene_object_id_key] = {
                            'instruction': scene_object_id_key.split('@')[1],
                            'pred': candidates[scene_object_id_key],
                            'gt': corpus[scene_object_id_key],
                        }
                    json.dump(pred_gt_val, f, indent=4)
                print(f'save pred_gt_val {curr_iter}')
        barrier()

    if is_primary():
        logout('\n----------------------Evaluation-----------------------\n')

        with open(os.path.join(args.checkpoint_dir, 'corpus_val.json'),
                  'w') as f:
            json.dump(corpus, f, indent=4)

        with open(os.path.join(args.checkpoint_dir, 'pred_val.json'),
                  'w') as f:
            json.dump(candidates, f, indent=4)

        with open(os.path.join(args.checkpoint_dir, 'qa_pred_gt_val.json'),
                  'w') as f:
            pred_gt_val = {}
            for index_, scene_object_id_key in enumerate(candidates):
                pred_gt_val[scene_object_id_key] = {
                    'instruction': scene_object_id_key.split('@')[1],
                    'pred': candidates[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                }

            json.dump(pred_gt_val, f, indent=4)
    # end of forward pass traversion
    metric_results = evaluator.start_evaluation()
    results_record = evaluator.records

    if is_primary():

        with open(
                os.path.join(args.checkpoint_dir,
                             'qa_pred_gt_val_with_scores.json'), 'w') as f:
            pred_gt_val = {}
            for index_, scene_object_id_key in enumerate(candidates):
                pred_gt_val[scene_object_id_key] = {
                    'instruction': scene_object_id_key.split('@')[1],
                    'pred': candidates[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                }
                pred_gt_val[scene_object_id_key].update(results_record[index_])
            json.dump(pred_gt_val, f, indent=4)
            json.dump(metric_results, f, indent=4)

    evaluator.reset()
    return metric_results
