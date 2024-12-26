import collections
import functools
import re
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import PartialState
from accelerate.utils import DistributedType, recursively_apply
from accelerate.utils.constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from torch._six import string_classes
from torch.utils.data import random_split

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

logger = get_logger(__name__)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def default_collate(batch):
    r"""
        Modify torch.utils.data.default_collate to support collating variable-length lists
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        'default_collate: batch must contain tensors, numpy arrays, numbers, '
        'dicts or lists; found {}')

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({
                key: default_collate([d[key] for d in batch])
                for key in elem
            })
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {
                key: default_collate([d[key] for d in batch])
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        """custom part: directly return for lists and tuples."""
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def split_train_set(train_set, epochs):
    train_subset_base = len(train_set) // epochs
    train_subset_res = len(train_set) % epochs
    return random_split(train_set,
                        [train_subset_base + 1] * (train_subset_res) +
                        [train_subset_base] * (epochs - train_subset_res))


# Customize operations for gathering
def _gpu_gather_object(object: Any):
    # by JY Huang: re-implement the method for gathering non-tensor objects
    output_objects = [None for _ in range(PartialState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    if isinstance(object, (list, tuple)):
        output_list = []
        for item in output_objects:
            output_list.extend(item)
        return output_list
    elif isinstance(object, dict):
        template = output_objects[0]
        output_dict = {}
        for k, v in template.items():
            if v is None or not hasattr(v, '__iter__'):
                output_dict[k] = v
                continue
            output_dict[k] = []
            for item in output_objects:
                output_dict[k].extend(item[k])
        return output_dict


def gather_object(object: Any):
    """Recursively gather object in a nested list/tuple/dictionary of objects
    from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if PartialState().distributed_type == DistributedType.TPU:
        raise NotImplementedError('gather objects in TPU is not supported')
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object


"""
Customize Accelerator to support:
    1. advanced gather_for_metrics
    2. only saving partial model weights when calling save_state
"""


class CustomAccelerator(Accelerator):

    def gather_for_metrics(self, input_data):
        # by JY Huang: re-implement this method for gathering non-tensor objects
        try:
            recursively_apply(lambda x: x,
                              input_data,
                              error_on_other_type=True)
            all_tensors = True
        except TypeError:
            all_tensors = False

        if not all_tensors:
            """custom part 1."""
            data = gather_object(input_data)
            """ custom part 1 """
        else:
            data = self.gather(input_data)

        try:
            if self.gradient_state.end_of_dataloader:
                # at the end of a dataloader, `gather_for_metrics` regresses to
                # `gather` unless the dataset has a remainder so log.
                if self.gradient_state.remainder == -1:
                    logger.info(
                        'The used dataset had no length, returning gathered tensors. You should drop the remainder yourself.'
                    )
                    return data
                elif self.gradient_state.remainder > 0:
                    """custom part 2."""

                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[:self.gradient_state.
                                      remainder] if tensor is not None else None

                    if all_tensors:
                        # This only applies to tensors, as defined in `recursively_apply`
                        return recursively_apply(_adjust_samples, data)
                    else:
                        if isinstance(data, (list, tuple)):
                            return _adjust_samples(data)
                        elif isinstance(data, dict):
                            return {
                                k: _adjust_samples(v)
                                for k, v in data.items()
                            }
                        else:
                            raise NotImplementedError(
                                f'Non-tensor gather only supports list, tuple or dict'
                            )
                    """ custom part 2 """
                else:  # remainder is 0
                    # no remainder even though at end of dataloader, so nothing to do.
                    return data
            else:
                # Not at the end of the dataloader, no need to adjust the tensors
                return data
        except Exception:
            # Dataset had no length or raised an error
            return data

    def get_state_dict(self, model, unwrap=True):
        # only save learnable parameters
        if self.distributed_type == DistributedType.DEEPSPEED:
            if self.deepspeed_config['zero_optimization']['stage'] == 3:
                if model.zero_gather_16bit_weights_on_model_save():
                    state_dict = model._zero3_consolidated_16bit_state_dict()
                else:
                    raise ValueError(
                        'Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. '
                        'To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or '
                        'set `zero3_save_16bit_model` to True when using `accelerate config`. '
                        'To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights.'
                    )
            else:
                from deepspeed.checkpoint.utils import \
                    clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(
                    self.unwrap_model(model).state_dict())
        elif self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True,
                                                         rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      full_state_dict_config):
                state_dict = model.state_dict()
        else:
            if unwrap:
                model = self.unwrap_model(model)
            state_dict = model.state_dict()
        """ custom part """
        keys_list = list(state_dict.keys())
        for k in keys_list:
            if k not in self.learn_params_list:  # need to assign `learn_params_list` before calling this method
                del state_dict[k]
        """ custom part """

        return state_dict

    def prepare_scheduler(self, scheduler: LRScheduler):
        # Ensure we can't double wrap a scheduler due to `find_batch_size`
        if getattr(scheduler, '_is_accelerate_prepared', False):
            if scheduler not in self._schedulers:
                self._schedulers.append(scheduler)
            return scheduler
        # We try to find the optimizer associated with `scheduler`, the default is the full list.
        optimizer = self._optimizers
        for opt in self._optimizers:
            if getattr(scheduler, 'optimizer', None) == opt.optimizer:
                optimizer = opt
                break
        scheduler = AcceleratedScheduler(
            scheduler,
            optimizer,
            step_with_optimizer=self.step_scheduler_with_optimizer,
            split_batches=True,  # custom, for proper scheduler.step()
        )
        self._schedulers.append(scheduler)
        return scheduler
