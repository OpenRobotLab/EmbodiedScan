import csv
import os

import numpy as np
from data.data_utils import VICUNA_ACTION_TOKENS
from transforms3d import euler

HABITAT_ACTION_SPACE = {
    'stop': 0,
    'move_forward': 1,
    'move_backward': 2,
    'turn_left': 3,
    'turn_right': 4,
    'look_up': 5,
    'look_down': 6,
    'grab_release': 7
}

HABITAT_ACTION_SPACE_REVERSE = {v: k for k, v in HABITAT_ACTION_SPACE.items()}

HABITAT_ACTION_SPACE_TOKENIZE = {
    k: v
    for k, v in zip(
        list(HABITAT_ACTION_SPACE.values()),
        list(VICUNA_ACTION_TOKENS.keys())[:len(HABITAT_ACTION_SPACE)])
}

HABITAT_ACTION_SPACE_DETOKENIZE = {
    v: k
    for k, v in zip(
        list(HABITAT_ACTION_SPACE.values()),
        list(VICUNA_ACTION_TOKENS.keys())[:len(HABITAT_ACTION_SPACE)])
}

shapenetcore_pp = [
    1, 6, 9, 11, 12, 14, 18, 23, 24, 25, 28, 31, 33, 35, 37, 41, 46, 48, 50,
    54, 56, 61, 65, 68, 72, 77, 80, 81, 82, 84, 87, 88, 93, 94, 96, 102, 107,
    112, 114, 115, 118, 125, 127, 130, 133, 141, 145, 150, 151, 159, 173, 175,
    194, 200, 206, 207, 208, 214, 215, 216, 226, 227, 229, 236, 238, 243, 246,
    247, 250, 259, 264, 266, 267, 269, 274, 279, 281, 288, 292, 294, 301, 306,
    308, 313, 322, 327, 331, 339, 358, 361, 366, 368, 387, 394, 396, 401, 402,
    404, 406, 407, 408, 419, 426, 442, 446, 452, 459, 461, 471, 473, 476, 483,
    487, 488, 490, 499, 501, 518, 529, 533, 540, 543, 554, 571, 576, 579, 580,
    589, 602, 608, 618, 621, 635, 639, 645, 660, 676, 677, 681, 683, 686, 694,
    697, 699, 709, 710, 713, 716, 728, 731, 738, 751, 755, 763, 768, 770, 783,
    795, 800, 804, 814, 815, 834, 838, 841, 844, 845, 851, 855, 856, 857, 863,
    870, 872, 874, 880, 888, 902, 915, 919, 931, 932, 943, 944, 946, 953, 956,
    967, 975, 983, 984, 985, 989, 990, 997, 1004, 1006, 1007, 1016, 1017, 1021,
    1038, 1062, 1066, 1074, 1080, 1093, 1102, 1115, 1119, 1121, 1122, 1124,
    1125, 1127, 1130, 1133, 1138, 1141, 1145, 1146, 1152, 1153, 1163, 1173,
    1184, 1192, 1193, 1194, 1198, 1208, 1211, 1214, 1226, 1238, 1246, 1247,
    1255, 1257, 1258, 1260, 1274, 1285, 1294, 1307, 1321, 1329, 1332, 1348,
    1350, 1351, 1376, 1378, 1398, 1400, 1403, 1408, 1409, 1415, 1417, 1423,
    1429, 1445, 1451, 1463, 1472, 1476, 1478, 1479, 1487, 1497, 1508, 1509,
    1510, 1518, 1528, 1537, 1540, 1543, 1549, 1551, 1563, 1565, 1569, 1582,
    1593, 1603, 1618, 1621, 1646, 1652, 1656
]


class CLIPortTokenizer:
    _BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    _PIXEL_SIZE = 0.003125
    _RESOLUTIN = np.array([320, 160])
    _ZBINS = 36
    _ZROT_EPSILLON = 0.

    def __init__(self) -> None:
        self.num_tokens_u = self._RESOLUTIN[0]
        self.num_tokens_v = self._RESOLUTIN[1]
        self.num_tokens_z = self._ZBINS

    def tokenize(self, act_pose: tuple) -> tuple:
        """Convert the action to a token."""
        act_trans = act_pose[0]
        act_quat = act_pose[1]
        ## limit the act_trans to the bounds
        act_trans[0] = min(max(act_trans[0], self._BOUNDS[0, 0]),
                           self._BOUNDS[0, 1])
        act_trans[1] = min(max(act_trans[1], self._BOUNDS[1, 0]),
                           self._BOUNDS[1, 1])

        u = int(
            np.round((act_trans[1] - self._BOUNDS[1, 0]) / self._PIXEL_SIZE))
        v = int(
            np.round((act_trans[0] - self._BOUNDS[0, 0]) / self._PIXEL_SIZE))
        ## set u, v to bound if out of bound
        u = min(max(u, 0), self._RESOLUTIN[0] - 1)
        v = min(max(v, 0), self._RESOLUTIN[1] - 1)
        ## quat to eulerXYZ
        quaternion_wxyz = np.array(
            [act_quat[3], act_quat[0], act_quat[1], act_quat[2]])
        euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
        z_rot_cont = euler_zxy[0]
        ## convert z_rot_cont to -pi ~ pi
        # z_rot_cont = z_rot_cont % (2 * np.pi) - np.pi
        # z_rot = int((z_rot_cont + self._ZROT_EPSILLON) / (2 * np.pi / self._ZBINS)) + (self._ZBINS // 2) - 1
        ## convert z_rot_cont to 0 ~ 2pi
        z_rot_cont = z_rot_cont % (2 * np.pi)
        z_rot = int(
            (z_rot_cont + self._ZROT_EPSILLON) / (2 * np.pi / self._ZBINS))

        ## convert to token id
        act_token = (u, v, z_rot)
        return act_token

    def detokenize(self, act_token: tuple, hmap=None) -> tuple:
        """Recover the action from the token."""
        u, v, z_rot = act_token
        if hmap is None:
            hmap = -np.ones((self._RESOLUTIN[0], self._RESOLUTIN[1]))
        x = self._BOUNDS[0, 0] + v * self._PIXEL_SIZE
        y = self._BOUNDS[1, 0] + u * self._PIXEL_SIZE
        z = self._BOUNDS[2, 0] + hmap[u, v]
        xyz = np.array([x, y, z])

        z_rot = z_rot - (self._ZBINS // 2)
        z_rot_cont = z_rot * (2 * np.pi / self._ZBINS)
        z_rot_cont = z_rot_cont + 0.5 * (2 * np.pi / self._ZBINS)
        z_rot_cont = z_rot_cont % (2 * np.pi) - np.pi
        quaternion_wxyz = euler.euler2quat(*(z_rot_cont, 0., 0.), axes='szxy')
        quaternion_xyzw = np.array([
            quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3],
            quaternion_wxyz[0]
        ])

        return (xyz, quaternion_xyzw)


_cliport_tokenizer = CLIPortTokenizer()

_CLIPORT_ACTION_SPACE_U = {
    k: v
    for k, v in zip(
        range(_cliport_tokenizer.num_tokens_u),
        list(VICUNA_ACTION_TOKENS.keys())
        [len(HABITAT_ACTION_SPACE):len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u])
}

_CLIPORT_ACTION_SPACE_U_REVERSE = {
    v: k
    for k, v in zip(
        range(_cliport_tokenizer.num_tokens_u),
        list(VICUNA_ACTION_TOKENS.keys())
        [len(HABITAT_ACTION_SPACE):len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u])
}

_CLIPORT_ACTION_SPACE_V = {
    k: v
    for k, v in zip(
        range(_cliport_tokenizer.num_tokens_v),
        list(VICUNA_ACTION_TOKENS.keys())
        [len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u:len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u + _cliport_tokenizer.num_tokens_v])
}

_CLIPORT_ACTION_SPACE_V_REVERSE = {
    v: k
    for k, v in zip(
        range(_cliport_tokenizer.num_tokens_v),
        list(VICUNA_ACTION_TOKENS.keys())
        [len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u:len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u + _cliport_tokenizer.num_tokens_v])
}

_CLIPORT_ACTION_SPACE_ZROT = {
    k: v
    for k, v in zip(
        range(_cliport_tokenizer.num_tokens_z),
        list(VICUNA_ACTION_TOKENS.keys())
        [len(HABITAT_ACTION_SPACE) + _cliport_tokenizer.num_tokens_u +
         _cliport_tokenizer.num_tokens_v:len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u + _cliport_tokenizer.num_tokens_v +
         _cliport_tokenizer.num_tokens_z])
}

_CLIPORT_ACTION_SPACE_ZROT_REVERSE = {
    v: k
    for k, v in zip(
        range(_cliport_tokenizer.num_tokens_z),
        list(VICUNA_ACTION_TOKENS.keys())
        [len(HABITAT_ACTION_SPACE) + _cliport_tokenizer.num_tokens_u +
         _cliport_tokenizer.num_tokens_v:len(HABITAT_ACTION_SPACE) +
         _cliport_tokenizer.num_tokens_u + _cliport_tokenizer.num_tokens_v +
         _cliport_tokenizer.num_tokens_z])
}

_DUMMY_CLIPORT_ACTION = {
    'pose0': (_cliport_tokenizer._BOUNDS[:, 0], np.array([0., 0., 0., 1.])),
    'pose1': (_cliport_tokenizer._BOUNDS[:, 0], np.array([0., 0., 0., 1.]))
}


def CLIPORT_ACTION_SPACE_TOKENIZE(action):
    global _cliport_tokenizer
    action_tokens = list(_cliport_tokenizer.tokenize(action))
    action_tokens[0] = _CLIPORT_ACTION_SPACE_U[action_tokens[0]]
    action_tokens[1] = _CLIPORT_ACTION_SPACE_V[action_tokens[1]]
    action_tokens[2] = _CLIPORT_ACTION_SPACE_ZROT[action_tokens[2]]
    return action_tokens


def CLIPORT_ACTION_SPACE_DETOKENIZE(token, obs=None):
    global _cliport_tokenizer
    u = _CLIPORT_ACTION_SPACE_U_REVERSE[token[0]]
    v = _CLIPORT_ACTION_SPACE_V_REVERSE[token[1]]
    z_rot = _CLIPORT_ACTION_SPACE_ZROT_REVERSE[token[2]]
    hmap = obs['depthmap'] if obs is not None else None
    return _cliport_tokenizer.detokenize((u, v, z_rot), hmap)


def _extract_between(lst, start, end, padding):
    # start and end are both included.

    # Calculate the desired output length
    length = end - start + 1

    # Pad the list at the beginning and end
    lst = [padding] * max(0, -start) + lst + [padding] * max(
        0, end - len(lst) + 1)

    # Adjust the start and end based on the padding
    start += max(0, -start)
    end = start + length - 1

    return lst[start:end + 1]


def filter_object_type(objects):
    obj_list = []
    for obj_pcd in objects:
        # TODO(jxma): we assume the last column is the semantic label
        sem = obj_pcd[0, -1] - 1
        if sem in shapenetcore_pp:
            obj_list.append(obj_pcd)
    return obj_list


def read_label_mapping(filename, label_from='category', label_to='index'):
    # mapping start from 0
    assert os.path.isfile(filename)
    mapping = dict()
    rmapping = dict()
    with open(filename) as tsvfile:
        tsvfile_content = tsvfile.read().replace('    ', '\t')
        reader = csv.DictReader(tsvfile_content.splitlines(), delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to]) - 1
            rmapping[int(row[label_to]) - 1] = row[label_from]
    return mapping, rmapping


def _process_object_feature(obj_list,
                            max_obj_points,
                            label_mapping_path=None,
                            habitat_alignment=False):
    # label_mapping_path:
    #   path to "matterport_category_mappings.tsv"
    #
    # - **all object pc will be centralized**
    # - you may provide a local seed to make this deterministic
    obj_list = np.array(obj_list, dtype=object)
    obj_fts = []
    obj_locs = []
    if label_mapping_path is not None:
        obj_sems = []
        _, mapping = read_label_mapping(label_mapping_path)
    for obj_pcd in obj_list:
        # TODO(jxma): Align obj pc with habitat coordinate
        if habitat_alignment:
            obj_pcd[:, 1], obj_pcd[:, 2] = obj_pcd[:, 2], -obj_pcd[:, 1]
        obj_center = obj_pcd[:, :3].mean(0)
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        obj_locs.append(np.concatenate([obj_center, obj_size], 0))
        pcd_idxs = np.random.choice(len(obj_pcd),
                                    size=max_obj_points,
                                    replace=(len(obj_pcd) < max_obj_points))
        obj_pcd = obj_pcd[pcd_idxs]

        # TODO(jxma): now we just centralized all object pc
        obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
        max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
        if max_dist < 1e-6:  # take care of tiny point-clouds, i.e., padding
            max_dist = 1
        obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist  # xyz normalize

        obj_pcd[:, 3:
                6] = obj_pcd[:, 3:
                             6] * 2 - 1  # rgb normalize (saved pointcloud have rgb as uint8 / 255.0)
        obj_fts.append(obj_pcd[:, :6])  # RGBXYZ
        if label_mapping_path is not None:
            # TODO(jxma): we assumt the last column is the semantic label
            sem = obj_pcd[0, -1] - 1  # from 1-1659 to 0-1658
            obj_sems.append(mapping[sem])

    obj_fts = np.stack(obj_fts, 0).astype(np.float32)
    obj_locs = np.array(obj_locs).astype(np.float32)
    obj_masks = np.ones(len(obj_locs)).astype(np.uint8)
    if label_mapping_path is not None:
        obj_sems = np.array(obj_sems).astype(object)
        return obj_fts, obj_locs, obj_masks, obj_sems
    else:
        return obj_fts, obj_locs, obj_masks


def prepare_object_feature_habitat(object_path, label_mapping_path,
                                   max_obj_points):
    # object_path:
    #   path to "objects.npy"
    # label_mapping_path:
    #   path to "matterport_category_mappings.tsv"
    #
    # - **all object pc will be centralized**
    # - you may provide a local seed to make this deterministic
    obj_list = np.load(object_path, allow_pickle=True)
    obj_list = filter_object_type(obj_list)
    return _process_object_feature(obj_list,
                                   max_obj_points,
                                   label_mapping_path,
                                   habitat_alignment=True)
