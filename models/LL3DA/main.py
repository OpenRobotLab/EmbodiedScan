import argparse
import importlib
import os
from collections import OrderedDict

import numpy as np
import torch
from datasets.scannet_base_dataset import DatasetConfig
from engine import do_train
from models.model_general import CaptionNet
from torch.multiprocessing import set_start_method
from utils.dist import (barrier, get_rank, init_distributed, is_distributed,
                        is_primary)
from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn


def make_args_parser():
    parser = argparse.ArgumentParser(
        'LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning',
        add_help=False)

    ##### Optimizer #####
    parser.add_argument('--base_lr', default=5e-4, type=float)
    parser.add_argument('--final_lr', default=1e-6, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--clip_gradient',
                        default=0.1,
                        type=float,
                        help='Max L2 norm of the gradient')
    # DISABLE warmup learning rate during dense caption training
    parser.add_argument('--warm_lr', default=1e-6, type=float)
    parser.add_argument('--warm_lr_epochs', default=9, type=int)
    # only ACTIVATE during dense caption training
    parser.add_argument('--pretrained_params_lr', default=None, type=float)
    parser.add_argument('--pretrained_weights', default=None, type=str)

    ##### Model #####
    # input based parameters
    parser.add_argument('--use_color', default=False, action='store_true')
    parser.add_argument('--use_normal', default=False, action='store_true')
    parser.add_argument('--no_height', default=False, action='store_true')
    parser.add_argument('--use_multiview', default=False, action='store_true')

    parser.add_argument('--detector',
                        default='detector_Vote2Cap_DETR',
                        help='folder of the detector')
    parser.add_argument('--captioner',
                        default=None,
                        type=str,
                        help='folder of the captioner')
    # training strategy
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
                        default=128,
                        type=int,
                        help='maximum length of object descriptions.')
    parser.add_argument('--max_gen_len',
                        default=32,
                        type=int,
                        help='maximum length of object descriptions.')

    ##### Dataset #####
    parser.add_argument('--max_prompts',
                        default=16,
                        type=int,
                        help='number of visual interactions')
    parser.add_argument('--dataset',
                        default='scannet',
                        help="dataset list split by ','")
    parser.add_argument('--grid_size_3d',
                        default=255,
                        type=int,
                        help='grid size of the 3D scene')
    parser.add_argument('--vocab',
                        default='llama-hf/7B',
                        type=str,
                        help='The LLM backend')
    parser.add_argument('--qformer_vocab',
                        default='bert-base-uncased',
                        type=str,
                        help='The QFormer backend')

    parser.add_argument('--dataset_num_workers', default=4, type=int)
    parser.add_argument('--batchsize_per_gpu', default=8, type=int)

    ##### Training #####
    parser.add_argument('--start_epoch', default=-1, type=int)
    parser.add_argument('--max_epoch', default=1080, type=int)
    parser.add_argument('--start_eval_after', default=-1, type=int)
    parser.add_argument('--eval_every_iteration', default=4000, type=int)
    parser.add_argument('--seed', default=0, type=int)

    ##### Testing #####
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument(
        '--test_min_iou',
        default=0.50,
        type=float,
        help='minimum iou for evaluating dense caption performance')
    parser.add_argument('--criterion',
                        default='CiDEr',
                        type=str,
                        help='metrics for saving the best model')
    parser.add_argument('--test_ckpt', default='', type=str)

    ##### I/O #####
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--save_every', default=4000, type=int)
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--filter_name',
                        default='captioner.transformer.',
                        type=str)

    ##### Distributed #####
    parser.add_argument('--ngpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--dist_url',
                        default='tcp://localhost:12345',
                        type=str)

    args = parser.parse_args()
    args.use_height = not args.no_height

    return args


def build_dataloader_func(args, dataset, split):
    if is_distributed():
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=(split == 'train'))

        print('Sample different samples for each GPUs.')
    else:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
    )
    return sampler, dataloader


def build_dataset(args):

    dataset_config = DatasetConfig()
    datasets = {'train': None, 'test': []}

    train_datasets = []
    for dataset in args.dataset.split(','):
        dataset_module = importlib.import_module(f'datasets.{dataset}')
        train_datasets.append(
            dataset_module.Dataset(args,
                                   dataset_config,
                                   split_set='train',
                                   use_color=args.use_color,
                                   use_normal=args.use_normal,
                                   use_multiview=args.use_multiview,
                                   use_height=args.use_height,
                                   augment=True))
        datasets['test'].append(
            dataset_module.Dataset(args,
                                   dataset_config,
                                   split_set='val',
                                   use_color=args.use_color,
                                   use_normal=args.use_normal,
                                   use_multiview=args.use_multiview,
                                   use_height=args.use_height,
                                   augment=False))
    datasets['train'] = torch.utils.data.ConcatDataset(train_datasets)

    train_sampler, train_loader = build_dataloader_func(args,
                                                        datasets['train'],
                                                        split='train')
    dataloaders = {
        'train': train_loader,
        'test': [],
        'train_sampler': train_sampler,
    }
    for dataset in datasets['test']:
        _, test_loader = build_dataloader_func(args, dataset, split='test')
        dataloaders['test'].append(test_loader)

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
    model = CaptionNet(args, dataset_config, datasets['train'])

    # testing phase
    if args.test_only:

        try:
            checkpoint = torch.load(args.test_ckpt,
                                    map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'], strict=False)
        except:
            print('test the model from scratch...')

        model_no_ddp = model.cuda()
        model = model.cuda(local_rank)

        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])

        for test_loader in dataloaders['test']:
            test_loader.dataset.eval_func(args, -1, model, dataset_config,
                                          test_loader)

    # training phase
    else:

        assert (args.checkpoint_dir is not None
                ), 'Please specify a checkpoint dir using --checkpoint_dir'
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        ### whether or not use pretrained weights
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights,
                                    map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)

            print('====                                          ====')
            print('==== loading following pre-trained parameters ====')
            print('====                                          ====')
            for name, param in checkpoint['model'].items():
                print('\t', name, param.shape)

        model_no_ddp = model.cuda()
        model = model.cuda(local_rank)

        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])

        if args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(filter(
                lambda params: params.requires_grad,
                model_no_ddp.parameters()),
                                          lr=args.base_lr,
                                          weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(filter(
                lambda params: params.requires_grad,
                model_no_ddp.parameters()),
                                        lr=args.base_lr,
                                        weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        print('====                                          ====')
        print('====  Only training the following parameters  ====')
        print('====                                          ====')
        for name, param in model_no_ddp.named_parameters():
            if param.requires_grad is True:
                print('\t', name, param.shape)

        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer)
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


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
