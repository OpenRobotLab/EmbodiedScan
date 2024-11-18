from fvcore.common.registry import Registry

MODULE_REGISTRY = Registry('Module')


def build_module(cfg):
    return MODULE_REGISTRY.get(cfg.name)(cfg)
