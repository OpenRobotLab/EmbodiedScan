import json
import os
import time
from collections import OrderedDict, defaultdict

import torch
import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.meteor.meteor as capmeteor
import utils.capeval.rouge.rouge as caprouge
from utils.box_util import box3d_iou_batch_tensor
from utils.dist import all_gather_dict, barrier, is_primary
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


def prepare_corpus(raw_data, max_len: int = 30) -> dict:
    # helper function to prepare ground truth captions
    corpus = defaultdict(list)
    object_id_to_name = defaultdict(lambda: 'unknown')

    for data in raw_data:

        (scene_id, object_id, object_name
         ) = data['scene_id'], data['object_id'], data['object_name']

        # parse language tokens
        token = data['token'][:max_len]
        description = ' '.join(['sos'] + token + ['eos'])
        key = f'{scene_id}|{object_id}|{object_name}'
        object_id_to_name[f'{scene_id}|{object_id}'] = object_name

        corpus[key].append(description)

    return corpus, object_id_to_name


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
    scene_list = dataset_loader.dataset.scan_names
    corpus, object_id_to_name = prepare_corpus(
        dataset_loader.dataset.scanrefer)
    task_name = dataset_loader.dataset.task_name
    ### initialize and prepare for evaluation
    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)

    model.eval()
    barrier()

    epoch_str = f'[{curr_epoch}/{args.max_epoch}]' if curr_epoch > 0 else ''

    candidates = {'caption': OrderedDict({}), 'iou': defaultdict(float)}

    for curr_iter, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
            'instruction': batch_data_label['instruction'],
            'instruction_mask': batch_data_label['instruction_mask'],
            'qformer_input_ids': batch_data_label['qformer_input_ids'],
            'qformer_attention_mask':
            batch_data_label['qformer_attention_mask'],
        }
        outputs = model(model_input, is_eval=True, task_name='dense-cap')

        outputs = dict(
            box_corners=outputs['box_corners'],
            sem_cls_prob=outputs['sem_cls_prob'],
            objectness_prob=outputs['objectness_prob'],
            output_ids=outputs['output_ids'],
            sem_cls_logits=outputs['sem_cls_logits'],
        )

        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)

        ### match objects
        batch_size, MAX_NUM_OBJ, _, _ = batch_data_label[
            'gt_box_corners'].shape
        _, nqueries, _, _ = outputs['box_corners'].shape

        match_box_ious = box3d_iou_batch_tensor(  # batch, nqueries, MAX_NUM_OBJ
            (outputs['box_corners'].unsqueeze(2).repeat(
                1, 1, MAX_NUM_OBJ, 1, 1).view(-1, 8, 3)),
            (batch_data_label['gt_box_corners'].unsqueeze(1).repeat(
                1, nqueries, 1, 1, 1).view(-1, 8, 3))).view(
                    batch_size, nqueries, MAX_NUM_OBJ)
        match_box_ious, match_box_idxs = match_box_ious.max(
            -1)  # batch, nqueries
        match_box_idxs = torch.gather(batch_data_label['gt_object_ids'], 1,
                                      match_box_idxs)  # batch, nqueries

        # ---- Checkout bounding box ious and semantic logits
        good_bbox_masks = match_box_ious > args.test_min_iou  # batch, nqueries
        good_bbox_masks &= outputs['sem_cls_logits'].argmax(-1) != (
            outputs['sem_cls_logits'].shape[-1] - 1)

        # ---- add nms to get accurate predictions
        nms_bbox_masks = parse_predictions(  # batch x nqueries
            outputs['box_corners'], outputs['sem_cls_prob'],
            outputs['objectness_prob'], batch_data_label['point_clouds'])
        nms_bbox_masks = torch.from_numpy(nms_bbox_masks).long() == 1
        good_bbox_masks &= nms_bbox_masks.to(good_bbox_masks.device)

        good_bbox_masks = good_bbox_masks.cpu().tolist()

        output_ids = outputs['output_ids']  # batch x nqueries x max_length
        captions = tokenizer.batch_decode(output_ids.reshape(
            -1, output_ids.shape[-1]),
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        captions = [
            [
                ('sos ' + captions[batch_id * nqueries + prop_id] + ' eos').replace('  ', ' ') \
                    for prop_id in range(nqueries)
            ] \
            for batch_id in range(batch_size)
        ]

        match_box_idxs = match_box_idxs.cpu().tolist()
        match_box_ious = match_box_ious.cpu().tolist()
        ### calculate measurable indicators on captions
        for idx, scene_id in enumerate(
                batch_data_label['scan_idx'].cpu().tolist()):
            scene_name = scene_list[scene_id]
            for prop_id in range(nqueries):

                if good_bbox_masks[idx][prop_id] is False:
                    continue

                match_obj_id = match_box_idxs[idx][prop_id]
                match_obj_iou = match_box_ious[idx][prop_id]

                object_name = object_id_to_name[f'{scene_name}|{match_obj_id}']
                key = f'{scene_name}|{match_obj_id}|{object_name}'

                if match_obj_iou > candidates['iou'][key]:
                    candidates['iou'][key] = match_obj_iou
                    candidates['caption'][key] = [captions[idx][prop_id]]
                    # DEBUG: checkout how many matched bounding boxes
                    # candidates[key] = ["this is a valid match!"]

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logout(f'Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; '
                   f'Evaluating on iter: {curr_train_iter}; '
                   f'Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB')
        barrier()
    # end of forward pass traversion

    ### message out
    missing_proposals = len(corpus.keys() - candidates['caption'].keys())
    total_captions = len(corpus.keys())

    ### make up placeholders for undetected bounding boxes
    for missing_key in (corpus.keys() - candidates['caption'].keys()):
        candidates['caption'][missing_key] = ['sos eos']

    # find annotated objects in scanrefer
    candidates = OrderedDict([
        (key, value) for key, value in sorted(candidates['caption'].items()) \
            if not key.endswith('unknown')
    ])
    score_per_caption, message, eval_metric = score_captions(
        OrderedDict([(key, corpus[key]) for key in candidates]), candidates)

    if is_primary():
        logout(f'\n----------------------Evaluation-----------------------\n'
               f'INFO: iou@{args.test_min_iou} matched proposals: '
               f'[{total_captions - missing_proposals} / {total_captions}], ')
        logout(message)

        with open(
                os.path.join(args.checkpoint_dir,
                             task_name + '_densecap_corpus_val.json'),
                'w') as f:
            json.dump(corpus, f, indent=4)

        with open(
                os.path.join(args.checkpoint_dir,
                             task_name + '_densecap_pred_val.json'), 'w') as f:
            json.dump(candidates, f, indent=4)

        with open(
                os.path.join(args.checkpoint_dir,
                             task_name + '_densecap_pred_gt_val.json'),
                'w') as f:
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

    eval_metrics = {
        metric + f'@{args.test_min_iou}': score \
            for metric, score in eval_metric.items()
    }
    return eval_metrics
