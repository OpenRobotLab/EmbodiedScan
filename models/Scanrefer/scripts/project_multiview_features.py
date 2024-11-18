import argparse
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.config import CONF
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


def get_scene_list():
    with open(SCANNET_LIST, 'r') as f:
        return sorted(list(set(f.read().splitlines())))


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


def get_scene_data(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        # load the original vertices, not the axis-aligned ones
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
            print('found {} mappings in {} points from frame {}'.format(
                indices_3ds[i][0], num_points, i))

    return indices_3ds, indices_2ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--maxpool',
                        action='store_true',
                        help='use max pooling to aggregate features \
         (use majority voting in label projection mode)')
    args = parser.parse_args()

    # setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    scene_list = get_scene_list()
    scene_data = get_scene_data(scene_list)
    with h5py.File(ENET_FEATURE_DATABASE, 'w', libver='latest') as database:
        print('projecting multiview features to point cloud...')
        for scene_id in scene_list:
            print('processing {}...'.format(scene_id))
            scene = scene_data[scene_id]
            # load frames
            frame_list = list(
                map(
                    lambda x: x.split('.')[0],
                    sorted(
                        os.listdir(SCANNET_FRAME_ROOT.format(
                            scene_id, 'color')))))
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
                                              '{}.png'.format(frame_id)),
                    [41, 32])
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
            # point_features = to_tensor(scene).new(scene.shape[0], 128).fill_(0)
            # for i, projection in enumerate(projections):
            #     frame_id = projection[0]
            #     projection_3d = projection[1]
            #     projection_2d = projection[2]
            #     feat = to_tensor(np.load(ENET_FEATURE_PATH.format(scene_id, frame_id)))
            #     proj_feat = PROJECTOR.project(feat, projection_3d, projection_2d, scene.shape[0]).transpose(1, 0)
            #     if i == 0:
            #         point_features = proj_feat
            #     else:
            #         mask = ((point_features == 0).sum(1) == 128).nonzero().squeeze(1)
            #         point_features[mask] = proj_feat[mask]

            # project
            point_features = to_tensor(scene).new(scene.shape[0], 128).fill_(0)
            for i, projection in enumerate(projections):
                frame_id = projection[0]
                projection_3d = projection[1]
                projection_2d = projection[2]
                feat = to_tensor(
                    np.load(ENET_FEATURE_PATH.format(scene_id, frame_id)))

                proj_feat = PROJECTOR.project(feat, projection_3d,
                                              projection_2d,
                                              scene.shape[0]).transpose(1, 0)

                if args.maxpool:
                    # only apply max pooling on the overlapping points
                    # find out the points that are covered in projection
                    feat_mask = ((proj_feat == 0).sum(1) != 128).bool()
                    # find out the points that are not filled with features
                    point_mask = ((point_features == 0).sum(1) == 128).bool()

                    # for the points that are not filled with features
                    # and are covered in projection,
                    # simply fill those points with projected features
                    mask = point_mask * feat_mask
                    point_features[mask] = proj_feat[mask]

                    # for the points that have already been filled with features
                    # and are covered in projection,
                    # apply max pooling first and then fill with pooled values
                    mask = ~point_mask * feat_mask
                    point_features[mask] = torch.max(point_features[mask],
                                                     proj_feat[mask])
                else:
                    if i == 0:
                        point_features = proj_feat
                    else:
                        mask = (point_features == 0).sum(1) == 128
                        point_features[mask] = proj_feat[mask]

            # save
            database.create_dataset(scene_id,
                                    data=point_features.cpu().numpy())

    print('done!')
