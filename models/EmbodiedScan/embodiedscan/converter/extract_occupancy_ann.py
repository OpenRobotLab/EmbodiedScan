import os
import shutil
from argparse import ArgumentParser

from tqdm import tqdm


def extract_occupancy(dataset, src, dst):
    """Extract occupancy annotations of a single dataset to dataset root."""
    print('Processing dataset', dataset)
    scenes = os.listdir(os.path.join(src, dataset))
    dst_dataset = os.path.join(dst, dataset)
    if not os.path.exists(dst_dataset):
        print('Missing dataset:', dataset)
        return
    for scene in tqdm(scenes):
        if dataset == 'scannet':
            dst_scene = os.path.join(dst_dataset, 'scans', scene)
        else:
            dst_scene = os.path.join(dst_dataset, scene)

        if not os.path.exists(dst_scene):
            print(f'Missing scene {scene} in dataset {dataset}')
            continue
        dst_occ = os.path.join(dst_scene, 'occupancy')
        if not os.path.exists(dst_occ):
            shutil.copytree(os.path.join(src, dataset, scene), dst_occ)
        else:
            files = os.listdir(os.path.join(src, dataset, scene))
            for file in files:
                if not os.path.exists(os.path.join(dst_occ, file)):
                    shutil.copyfile(os.path.join(src, dataset, scene, file),
                                    os.path.join(dst_occ, file))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src',
                        required=True,
                        help='folder of the occupancy annotations')
    parser.add_argument('--dst',
                        required=True,
                        help='folder of the raw datasets')
    args = parser.parse_args()
    datasets = os.listdir(args.src)
    for dataset in datasets:
        extract_occupancy(dataset, args.src, args.dst)
