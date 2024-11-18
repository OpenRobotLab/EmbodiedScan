import argparse
import math
import os
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.config import CONF
from lib.enet import create_enet_for_3d
from lib.projection import ProjectionHelper

SCANNET_LIST = CONF.SCANNETV2_LIST
SCANNET_DATA = CONF.PATH.SCANNET_DATA
SCANNET_FRAME_ROOT = CONF.SCANNET_FRAMES
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, '{}')  # name of the file

ENET_FEATURE_PATH = CONF.ENET_FEATURES_PATH
ENET_FEATURE_DATABASE = CONF.MULTIVIEW

# projection
INTRINSICS = [[37.01983, 0, 20, 0], [0, 38.52470, 15.5, 0], [0, 0, 1, 0],
              [0, 0, 0, 1]]
PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [41, 32], 0.05)

ENET_PATH = CONF.ENET_WEIGHTS
ENET_GT_PATH = SCANNET_FRAME_PATH

NYU40_LABELS = CONF.NYU40_LABELS
SCANNET_LABELS = [
    'unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed',
    'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter',
    'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet',
    'otherfurniture'
]

PC_LABEL_ROOT = os.path.join(CONF.PATH.OUTPUT, 'projections')
PC_LABEL_PATH = os.path.join(PC_LABEL_ROOT, '{}.ply')


def get_nyu40_labels():
    labels = ['unannotated']
    labels += pd.read_csv(NYU40_LABELS)['nyu40class'].tolist()

    return labels


def get_prediction_to_raw():
    labels = get_nyu40_labels()
    mapping = {i: label for i, label in enumerate(labels)}

    return mapping


def get_nyu_to_scannet():
    nyu_idx_to_nyu_label = get_prediction_to_raw()
    scannet_label_to_scannet_idx = {
        label: i
        for i, label in enumerate(SCANNET_LABELS)
    }

    # mapping
    nyu_to_scannet = {}
    for nyu_idx in range(41):
        nyu_label = nyu_idx_to_nyu_label[nyu_idx]
        if nyu_label in scannet_label_to_scannet_idx.keys():
            scannet_idx = scannet_label_to_scannet_idx[nyu_label]
        else:
            scannet_idx = 0
        nyu_to_scannet[nyu_idx] = scannet_idx

    return nyu_to_scannet


def create_color_palette():
    return {
        'unannotated': (0, 0, 0),
        'floor': (152, 223, 138),
        'wall': (174, 199, 232),
        'cabinet': (31, 119, 180),
        'bed': (255, 187, 120),
        'chair': (188, 189, 34),
        'sofa': (140, 86, 75),
        'table': (255, 152, 150),
        'door': (214, 39, 40),
        'window': (197, 176, 213),
        'bookshelf': (148, 103, 189),
        'picture': (196, 156, 148),
        'counter': (23, 190, 207),
        'desk': (247, 182, 210),
        'curtain': (219, 219, 141),
        'refridgerator': (255, 127, 14),
        'bathtub': (227, 119, 194),
        'shower curtain': (158, 218, 229),
        'toilet': (44, 160, 44),
        'sink': (112, 128, 144),
        'otherfurniture': (82, 84, 163),
    }


def get_scene_list(args):
    if args.scene_id == '-1':
        with open(SCANNET_LIST, 'r') as f:
            return sorted(list(set(f.read().splitlines())))
    else:
        return [args.scene_id]


def to_tensor(arr):
    return torch.Tensor(arr).cuda()


def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(
        math.floor(new_image_dims[1] * float(image_dims[0]) /
                   float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width],
                              interpolation=Image.NEAREST)(
                                  Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1],
                                   new_image_dims[0]])(image)
    image = np.array(image)

    return image


def load_image(file, image_dims):
    image = imread(file)
    # preprocess
    image = resize_crop_image(image, image_dims)
    if len(image.shape) == 3:  # color image
        image = np.transpose(image, [2, 0, 1])  # move feature to front
        image = transforms.Normalize(
            mean=[0.496342, 0.466664, 0.440796],
            std=[0.277856, 0.28623,
                 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
    elif len(image.shape) == 2:  # label image
        #         image = np.expand_dims(image, 0)
        pass
    else:
        raise

    return image


def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(' ') for x in lines)]

    return np.asarray(lines).astype(np.float32)


def load_depth(file, image_dims):
    depth_image = imread(file)
    # preprocess
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0

    return depth_image


def visualize(coords, labels):
    palette = create_color_palette()
    nyu_to_scannet = get_nyu_to_scannet()
    vertex = []
    for i in range(coords.shape[0]):
        vertex.append((coords[i][0], coords[i][1], coords[i][2],
                       palette[SCANNET_LABELS[nyu_to_scannet[labels[i]]]][0],
                       palette[SCANNET_LABELS[nyu_to_scannet[labels[i]]]][1],
                       palette[SCANNET_LABELS[nyu_to_scannet[labels[i]]]][2]))

    vertex = np.array(vertex,
                      dtype=[('x', np.dtype('float32')),
                             ('y', np.dtype('float32')),
                             ('z', np.dtype('float32')),
                             ('red', np.dtype('uint8')),
                             ('green', np.dtype('uint8')),
                             ('blue', np.dtype('uint8'))])

    output_pc = PlyElement.describe(vertex, 'vertex')
    output_pc = PlyData([output_pc])
    os.makedirs(PC_LABEL_ROOT, exist_ok=True)
    output_pc.write(PC_LABEL_PATH.format(args.scene_id))


def get_scene_data(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        scene_data[scene_id] = {}
        scene_data[scene_id] = np.load(
            os.path.join(SCANNET_DATA, scene_id) + '_vert.npy')[:, :3]

    return scene_data


def compute_projection(points, depth, camera_to_world):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)

        :return indices_3d (array with point indices that correspond to a pixel),
        :return indices_2d (array with pixel indices that correspond to a point)

        note:
            the first digit of indices represents the number of relevant points
            the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(to_tensor(points),
                                               to_tensor(depth[i]),
                                               to_tensor(camera_to_world[i]))
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()

    return indices_3ds, indices_2ds


def create_enet():
    enet_fixed, enet_trainable, enet_classifier = create_enet_for_3d(
        41, ENET_PATH, 21)
    enet = nn.Sequential(enet_fixed, enet_trainable, enet_classifier).cuda()
    enet.eval()
    for param in enet.parameters():
        param.requires_grad = False

    return enet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, default='-1')
    parser.add_argument('--gt', action='store_true')
    parser.add_argument('--maxpool',
                        action='store_true',
                        help='use max pooling to aggregate features \
         (use majority voting in label projection mode)')
    args = parser.parse_args()

    scene_list = get_scene_list(args)
    scene_data = get_scene_data(scene_list)
    enet = create_enet()
    for scene_id in tqdm(scene_list):
        scene = scene_data[scene_id]
        # load frames
        frame_list = list(
            map(
                lambda x: x.split('.')[0],
                sorted(os.listdir(SCANNET_FRAME_ROOT.format(scene_id,
                                                            'color')))))
        scene_images = np.zeros((len(frame_list), 3, 256, 328))
        scene_depths = np.zeros((len(frame_list), 32, 41))
        scene_poses = np.zeros((len(frame_list), 4, 4))
        for i, frame_id in enumerate(frame_list):
            scene_images[i] = load_image(
                SCANNET_FRAME_PATH.format(scene_id, 'color',
                                          '{}.jpg'.format(frame_id)),
                [328, 256])
            scene_depths[i] = load_depth(
                SCANNET_FRAME_PATH.format(scene_id, 'depth',
                                          '{}.png'.format(frame_id)), [41, 32])
            scene_poses[i] = load_pose(
                SCANNET_FRAME_PATH.format(scene_id, 'pose',
                                          '{}.txt'.format(frame_id)))

        # compute projections for each chunk
        projection_3d, projection_2d = compute_projection(
            scene, scene_depths, scene_poses)

        # compute valid projections
        projections = []
        for i in range(projection_3d.shape[0]):
            num_valid = projection_3d[i, 0]
            if num_valid == 0:
                continue

            projections.append(
                (frame_list[i], projection_3d[i], projection_2d[i]))

        # # project
        # labels = None
        # for i, projection in enumerate(projections):
        #     frame_id = projection[0]
        #     projection_3d = projection[1]
        #     projection_2d = projection[2]
        #     if args.gt:
        #         feat = to_tensor(load_image(ENET_GT_PATH.format(scene_id, "labelv2", "{}.png".format(frame_id)), [41, 32])).unsqueeze(0)
        #     else:
        #         image = load_image(SCANNET_FRAME_PATH.format(scene_id, "color", "{}.jpg".format(frame_id)), [328, 256])
        #         feat = enet(to_tensor(image).unsqueeze(0)).max(1)[1].unsqueeze(1)

        #     proj_label = PROJECTOR.project(feat, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0)
        #     if i == 0:
        #         labels = proj_label
        #     else:
        #         labels[labels == 0] = proj_label[labels == 0]

        # project
        labels = to_tensor(scene).new(scene.shape[0],
                                      len(projections)).fill_(0).long()
        for i, projection in enumerate(projections):
            frame_id = projection[0]
            projection_3d = projection[1]
            projection_2d = projection[2]

            if args.gt:
                feat = to_tensor(
                    load_image(
                        ENET_GT_PATH.format(scene_id, 'labelv2',
                                            '{}.png'.format(frame_id)),
                        [41, 32])).unsqueeze(0)
            else:
                image = load_image(
                    SCANNET_FRAME_PATH.format(scene_id, 'color',
                                              '{}.jpg'.format(frame_id)),
                    [328, 256])
                feat = enet(
                    to_tensor(image).unsqueeze(0)).max(1)[1].unsqueeze(1)

            proj_label = PROJECTOR.project(feat, projection_3d, projection_2d,
                                           scene.shape[0]).transpose(
                                               1, 0)  # num_points, 1

            if args.maxpool:
                # only apply max pooling on the overlapping points
                # find out the points that are covered in projection
                feat_mask = ((proj_label == 0).sum(1) != 1).bool()
                # find out the points that are not filled with labels
                point_mask = ((labels == 0).sum(1) == len(projections)).bool()

                # for the points that are not filled with features
                # and are covered in projection,
                # simply fill those points with labels
                mask = point_mask * feat_mask
                labels[mask, i] = proj_label[mask, 0]

                # for the points that have already been filled with features
                # and are covered in projection,
                # simply fill those points with labels
                mask = ~point_mask * feat_mask
                labels[mask, i] = proj_label[mask, 0]
            else:
                if i == 0:
                    labels = proj_label
                else:
                    labels[labels == 0] = proj_label[labels == 0]

        # aggregate
        if args.maxpool:
            new_labels = []
            for label_id in range(labels.shape[0]):
                point_label = labels[label_id].cpu().numpy().tolist()
                count = dict(Counter(point_label))
                count = sorted(count.items(), key=lambda x: x[1], reverse=True)
                count = [c for c in count if c[0] != 0]
                if count:
                    new_labels.append(count[0][0])
                else:
                    new_labels.append(0)

            labels = torch.FloatTensor(np.array(new_labels)[:, np.newaxis])

        # output
        visualize(scene, labels.long().squeeze(1).cpu().numpy())
