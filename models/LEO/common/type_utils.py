import torch
from omegaconf import OmegaConf


def cfg2dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def _to_device(state, device):
    """usually load from cpu checkpoint but need to load to cuda."""
    if isinstance(state, torch.Tensor):
        new_state = state.to(
            device,
            non_blocking=True)  # assume propoerly set py torch.cuda.set_device
    elif isinstance(state, list):
        new_state = torch.tensor([_to_device(t, device)
                                  for t in state]).to(device)
    elif isinstance(state, tuple):
        new_state = torch.tensor(tuple(_to_device(t, device)
                                       for t in state)).to(device)
    elif isinstance(state, dict):
        new_state = {n: _to_device(t, device) for n, t in state.items()}
    else:
        try:
            if not isinstance(state, str):
                new_state = torch.tensor(state).to(device)
            else:
                new_state = state
        except:
            raise ValueError(
                f'The provided tensor can not be transfered to {device}')
    return new_state
