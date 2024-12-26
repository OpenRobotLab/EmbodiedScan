import time

import torch
from utils.ap_calculator import APCalculator
from utils.dist import all_gather_dict, barrier, is_primary
from utils.misc import SmoothedValue


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

    # ap calculator is exact for evaluation.
    #   This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

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
        }
        outputs = model(model_input, is_eval=True)

        outputs = dict(
            box_corners=outputs['box_corners'],
            sem_cls_prob=outputs['sem_cls_prob'],
            objectness_prob=outputs['objectness_prob'],
            point_clouds=batch_data_label['point_clouds'],
            gt_box_corners=batch_data_label['gt_box_corners'],
            gt_box_sem_cls_label=batch_data_label['gt_box_sem_cls_label'],
            gt_box_present=batch_data_label['gt_box_present'],
        )
        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        ap_calculator.step_meter({'outputs': outputs}, batch_data_label)
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logout(f'Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; '
                   f'Evaluating on iter: {curr_train_iter}; '
                   f'Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB')
        barrier()
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)

    if is_primary():
        logout('==' * 10)
        logout(f'Evaluate Epoch [{curr_epoch}/{args.max_epoch}]')
        logout(f'{metric_str}')
        logout('==' * 10)

    eval_metrics = {
        metric + f'@{args.test_min_iou}': score \
            for metric, score in metrics[args.test_min_iou].items()
    }
    return eval_metrics
