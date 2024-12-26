import argparse
import os

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id',
                        type=str,
                        help='scene id of scene to be visualized',
                        default='scene0000_00')
    args = parser.parse_args()

    verts = np.load('scannet_data/{}_vert.npy'.format(args.scene_id))
    aligned_verts = np.load('scannet_data/{}_aligned_vert.npy'.format(
        args.scene_id))

    with open('scannet_data/{}_verts.obj'.format(args.scene_id), 'w') as f:
        for i in range(verts.shape[0]):
            f.write('v {} {} {} {} {} {}\n'.format(verts[i, 0], verts[i, 1],
                                                   verts[i, 2], verts[i, 3],
                                                   verts[i, 4], verts[i, 5]))

    with open('scannet_data/{}_aligned_verts.obj'.format(args.scene_id),
              'w') as f:
        for i in range(aligned_verts.shape[0]):
            f.write('v {} {} {} {} {} {}\n'.format(
                aligned_verts[i, 0], aligned_verts[i, 1], aligned_verts[i, 2],
                aligned_verts[i, 3], aligned_verts[i, 4], aligned_verts[i, 5]))
