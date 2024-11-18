from common.misc import default_collate
from fvcore.common.registry import Registry
from torch.utils.data import DataLoader

DATASET_REGISTRY = Registry('Dataset')
DATASETWRAPPER_REGISTRY = Registry('DatasetWrapper')


def get_dataset_leo(cfg, split, dataset_name, dataset_wrapper_name,
                    dataset_wrapper_args):
    # just get dataset directly and then wrap it
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)

    if dataset_wrapper_name:
        dataset = DATASETWRAPPER_REGISTRY.get(dataset_wrapper_name)(
            dataset, dataset_wrapper_args)

    return dataset


def build_dataloader_leo(cfg, split, dataset_name, dataset_wrapper_name,
                         dataset_wrapper_args, dataloader_args):
    dataset = get_dataset_leo(cfg, split, dataset_name, dataset_wrapper_name,
                              dataset_wrapper_args)
    return DataLoader(dataset,
                      batch_size=dataloader_args.batchsize,
                      num_workers=dataloader_args.num_workers,
                      collate_fn=getattr(dataset, 'collate_fn',
                                         default_collate),
                      pin_memory=True,
                      shuffle=True if split == 'train' else False,
                      drop_last=True if split == 'train' else False)
