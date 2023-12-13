import os
import zipfile
from argparse import ArgumentParser
from functools import partial

import mmengine


def process_scene(path, scene_name):
    """Process single 3Rscan scene."""
    with zipfile.ZipFile(os.path.join(path, scene_name, 'sequence.zip'),
                         'r') as zip_ref:
        zip_ref.extractall(os.path.join(path, scene_name, 'sequence'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_folder',
                        required=True,
                        help='folder of the dataset.')
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    mmengine.track_parallel_progress(func=partial(process_scene,
                                                  args.dataset_folder),
                                     tasks=os.listdir(args.dataset_folder),
                                     nproc=args.nproc)
