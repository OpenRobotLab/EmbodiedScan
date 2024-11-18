import argparse
import importlib
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from engine import do_train
from models.model_general import CaptionNet
from torch.multiprocessing import set_start_method
from utils.dist import (all_gather_dict, barrier, get_rank, init_distributed,
                        is_distributed, is_primary)
from utils.io import resume_if_possible
from utils.misc import SmoothedValue, my_worker_init_fn


def make_args_parser():
    parser = argparse.ArgumentParser(
        'End-to-End 3D Dense Captioning with Vote2Cap-DETR', add_help=False)

    ##### Model #####
    # input based parameters
    parser.add_argument('--use_color', default=False, action='store_true')
    parser.add_argument('--use_normal', default=False, action='store_true')
    parser.add_argument('--no_height', default=False, action='store_true')
    parser.add_argument('--use_multiview', default=False, action='store_true')
    parser.add_argument('--max_prompts',
                        default=16,
                        type=int,
                        help='number of visual interactions')
    parser.add_argument('--grid_size_3d',
                        default=255,
                        type=int,
                        help='grid size of 3D environ')

    parser.add_argument('--detector',
                        default='detector_Vote2Cap_DETR',
                        choices=['detector_votenet', 'detector_Vote2Cap_DETR'],
                        help='folder of the detector')
    parser.add_argument('--captioner',
                        default=None,
                        type=str,
                        help='folder of the captioner')
    parser.add_argument(
        '--freeze_detector',
        default=False,
        action='store_true',
        help='freeze all parameters other than the caption head')
    parser.add_argument('--freeze_llm',
                        default=False,
                        action='store_true',
                        help='freeze the llm for caption generation')

    # caption related hyper parameters
    parser.add_argument(
        '--use_beam_search',
        default=False,
        action='store_true',
        help='whether use beam search during caption generation.')
    parser.add_argument('--max_des_len',
                        default=32,
                        type=int,
                        help='maximum length of object descriptions.')

    ##### Dataset #####
    parser.add_argument(
        '--dataset',
        default='scannet',
        help='dataset file which stores `dataset` and `dataset_config` class',
    )
    parser.add_argument('--vocab',
                        default='facebook/opt-1.3b',
                        type=str,
                        help='should be one of `gpt2` or `scanrefer`')
    parser.add_argument('--qformer_vocab',
                        default='bert-base-embedding',
                        type=str,
                        help='should be one of `gpt2` or `scanrefer`')
    parser.add_argument('--dataset_num_workers', default=4, type=int)
    parser.add_argument('--batchsize_per_gpu', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)

    ##### Testing #####
    parser.add_argument('--test_ckpt', default='', type=str)

    ##### I/O #####
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--log_every', default=10, type=int)

    ##### Distributed #####
    parser.add_argument('--ngpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--dist_url',
                        default='tcp://localhost:12345',
                        type=str)

    args = parser.parse_args()
    args.use_height = not args.no_height

    return args


@torch.no_grad()
def evaluate(
    args,
    task_name,
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
    candidates = []
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
        outputs = model(model_input, is_eval=True, task_name='qa')

        outputs = dict(output_ids=outputs['output_ids'], )

        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)

        output_ids = outputs['output_ids']  # batch x max_length
        answers = tokenizer.batch_decode(output_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

        quesition_index = batch_data_label['scan_idx'].reshape(-1)
        quesition_index = quesition_index.cpu().tolist()

        for idx in range(output_ids.shape[0]):
            anno = annotations[quesition_index[idx]]
            key = anno['question_id']
            answer = answers[idx]
            answer = ' '.join(filter(lambda w: w, answer.split(' ')))
            top10_answer = [answer for _ in range(10)]

            candidates.append({
                'scene_id': anno['scene_id'],
                'question_id': key,
                'answer_top10': top10_answer,
                'bbox': []
            })

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logout(f'Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; '
                   f'Evaluating on iter: {curr_train_iter}; '
                   f'Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB')
        barrier()

    # end of forward pass traversion
    if is_primary():
        with open(os.path.join(args.checkpoint_dir, f'{task_name}.json'),
                  'w') as f:
            json.dump(candidates, f, indent=4)

    return None


def build_dataset(args):
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    dataset_config = dataset_module.DatasetConfig()

    datasets = {
        'train':
        dataset_module.Dataset(args,
                               dataset_config,
                               split_set='train',
                               use_color=args.use_color,
                               use_normal=args.use_normal,
                               use_multiview=args.use_multiview,
                               use_height=args.use_height,
                               augment=True),
        'val':
        dataset_module.Dataset(args,
                               dataset_config,
                               split_set='val',
                               use_color=args.use_color,
                               use_normal=args.use_normal,
                               use_multiview=args.use_multiview,
                               use_height=args.use_height,
                               augment=False),
        'test_w_obj':
        dataset_module.Dataset(args,
                               dataset_config,
                               split_set='test_w_obj',
                               use_color=args.use_color,
                               use_normal=args.use_normal,
                               use_multiview=args.use_multiview,
                               use_height=args.use_height,
                               augment=False),
        'test_wo_obj':
        dataset_module.Dataset(args,
                               dataset_config,
                               split_set='test_wo_obj',
                               use_color=args.use_color,
                               use_normal=args.use_normal,
                               use_multiview=args.use_multiview,
                               use_height=args.use_height,
                               augment=False),
    }

    dataloaders = {}
    for split in datasets.keys():
        if is_distributed():
            sampler = torch.utils.data.DistributedSampler(
                datasets[split], shuffle=(split == 'train'))
        else:
            if split == 'train':
                sampler = torch.utils.data.RandomSampler(datasets[split])
            else:
                sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + '_sampler'] = sampler

    return dataset_config, datasets, dataloaders


def main(local_rank, args):

    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend='nccl',
        )

    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + get_rank())

    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError(
            'Either checkpoint_dir or test_ckpt should be presented!')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = CaptionNet(args, dataset_config, datasets['train']).cuda()

    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])

    # testing phase
    checkpoint = torch.load(args.test_ckpt, map_location=torch.device('cpu'))
    model_no_ddp.load_state_dict(checkpoint['model'], strict=False)

    evaluate(args, 'val', -1, model, dataset_config, dataloaders['val'])

    evaluate(args, 'test_wo_obj', -1, model, dataset_config,
             dataloaders['test_wo_obj'])

    evaluate(args, 'test_w_obj', -1, model, dataset_config,
             dataloaders['test_w_obj'])
    return


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args, ))


if __name__ == '__main__':
    args = make_args_parser()

    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    launch_distributed(args)
