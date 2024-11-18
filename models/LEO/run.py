import os
from datetime import datetime

import common.io_utils as iu
import hydra
from accelerate.logging import get_logger
from common.misc import rgetattr
from trainer.build import build_trainer

logger = get_logger(__name__)


@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg):
    os.environ[
        'TOKENIZERS_PARALLELISM'] = 'true'  # suppress hf tokenizer warning
    print(cfg.num_gpu)
    naming_keys = [cfg.name]
    for name in cfg.naming_keywords:
        key = str(rgetattr(cfg, name))
        if key:
            naming_keys.append(key)
    exp_name = '_'.join(naming_keys)

    # Record the experiment
    cfg.exp_dir = os.path.join(
        cfg.base_dir, exp_name,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        if 'time' in cfg.naming_keywords else '')
    iu.make_dir(cfg.exp_dir)
    iu.save_yaml(cfg, os.path.join(cfg.exp_dir, 'config.json'))

    trainer = build_trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
