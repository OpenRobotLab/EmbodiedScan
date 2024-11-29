import numpy as np
import json
import os
import cv2
from tqdm import tqdm

EXCLUDED_OBJECTS = ["wall", "ceiling", "floor"]


def reverse_multi2multi_mapping(mapping):
    """
    Args:
        mapping: dict in format key1:[value1, value2], key2:[value2, value3]
    Returns:
        mapping: dict in format value1:[key1], value2:[key1, key2], value3:[key2]
    """
    output = {}
    possible_values = []
    for key, values in mapping.items():
        for value in values:
            possible_values.append(value)
    possible_values = list(set(possible_values))
    for value in possible_values:
        output[value] = []
    for key, values in mapping.items():
        for value in values:
            output[value].append(key)
    return output


def reverse_121_mapping(mapping):
    """Reverse a 1-to-1 mapping.

    Args:
        mapping: dict in format key1:value1, key2:value2
    Returns:
        mapping: dict in format value1:key1, value2:key2
    """
    return {v: k for k, v in mapping.items()}


def load_json(path):
    if os.path.getsize(path) == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_extrinsic_dir(directory):
    """
    Returns:
        extrinsics: numpy array of extrinsic matrices, shape (N, 4, 4)
        ids: list of ids (str) of matrix files.
    """
    extrinsics = []
    ids = []
    for file in os.listdir(directory):
        if file.endswith(".txt") or file.endswith(".npy"):
            if file.startswith("depth_intrinsic") or file.startswith("intrinsic"):
                continue
            path = os.path.join(directory, file)
            extrinsics.append(read_extrinsic(path))
            path = path.replace("\\", "/")
            ids.append(file.split(".")[0])
    return extrinsics, ids


def _pad_extrinsic(mat):
    """transforms the extrinsic matrix to the 4x4 form."""
    mat = np.array(mat)
    if mat.shape == (3, 4):
        mat = np.vstack((mat, [0, 0, 0, 1]))
    elif mat.shape != (4, 4):
        raise ValueError("Invalid shape of matrix.")
    return mat


def read_extrinsic(path):
    """returns a 4x4 numpy array of intrinsic matrix."""
    if path.endswith(".txt"):
        mat = np.loadtxt(path)
        return _pad_extrinsic(mat)
    elif path.endswith(".npy"):
        mat = np.load(path)
        return _pad_extrinsic(mat)
    else:
        raise ValueError("Invalid file extension.")


def _read_intrinsic_mp3d(path):
    a = np.loadtxt(path)
    intrinsic = np.identity(4, dtype=float)
    intrinsic[0][0] = a[2]  # fx
    intrinsic[1][1] = a[3]  # fy
    intrinsic[0][2] = a[4]  # cx
    intrinsic[1][2] = a[5]  # cy
    # a[0], a[1] are the width and height of the image
    return intrinsic


def _read_intrinsic_scannet(path):
    intrinsic = np.loadtxt(path)
    return intrinsic


def read_intrinsic(path, mode="scannet"):
    """Reads intrinsic matrix from file.

    Returns:
        extended intrinsic of shape (4, 4)
    """
    if mode == "scannet":
        return _read_intrinsic_scannet(path)
    elif mode == "mp3d":
        return _read_intrinsic_mp3d(path)
    else:
        raise ValueError("Invalid mode {}.".format(mode)) 

def _read_axis_align_matrix_scannet(path):
    with open(path, 'r') as file:
        first_line = file.readline()
    vals = first_line.strip().split(' ')[2:]
    vals = np.array(vals, dtype=np.float64)
    output = vals.reshape(4, 4)
    return output

def read_axis_align_matrix(path, mode):
    if mode == "scannet":
        return _read_axis_align_matrix_scannet(path)
    else:
        raise ValueError("Invalid mode {}.".format(mode)) 

def read_depth_map(path):
    """Reads depth map from file.

    Returns:
        depth: numpy array of depth values, shape (H, W)
    """
    if "3rscan" in path:
        path = path[:-4] + ".pgm"
    depth_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Cannot read file {path}")
    depth_map = depth_map/1000.0 # '/=' does not work. Interesting.
    if "matterport" in path or "mp3d" in path:
        depth_map /= 4.0  # for matterport, depth should be divided by 4000
    return depth_map


def read_bboxes_json(path, return_id=False, return_type=False):
    """
    Returns:
        boxes: numpy array of bounding boxes, shape (M, 9): xyz, lwh, ypr
        ids: (optional) numpy array of obj ids, shape (M,)
        types: (optional) list of strings, each string is a type of object
    """
    with open(path, "r") as f:
        bboxes_json = json.load(f)
    boxes = []
    ids = []
    types = []
    for i in range(len(bboxes_json)):
        if bboxes_json[i]["obj_type"] in EXCLUDED_OBJECTS:
            continue
        box = bboxes_json[i]["psr"]
        position = np.array(
            [box["position"]["x"], box["position"]["y"], box["position"]["z"]]
        )
        size = np.array([box["scale"]["x"], box["scale"]["y"], box["scale"]["z"]])
        euler_angles = np.array(
            [box["rotation"]["x"], box["rotation"]["y"], box["rotation"]["z"]]
        )
        boxes.append(np.concatenate([position, size, euler_angles]))
        ids.append(int(bboxes_json[i]["obj_id"]))
        types.append(bboxes_json[i]["obj_type"])
    boxes = np.array(boxes)
    if return_id and return_type:
        ids = np.array(ids)
        return boxes, ids, types
    if return_id:
        ids = np.array(ids)
        return boxes, ids
    if return_type:
        return boxes, types
    return boxes


def get_scene_prefix(path):
    if "3rscan" in path:
        return "3rscan"
    elif "matterport" in path or "mp3d" in path:
        return "matterport3d"
    elif "scene" in path:
        return "scannet"
    else:
        return ""

def read_type2int(path):
    with open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
    metainfo = data["metainfo"]
    object_type_to_int = metainfo["categories"]
    return object_type_to_int

def apply_mapping_to_keys(d, mappings):
    """
    Args:
        d: a dictionary
        mappings: dictionary(s) of mappings, e.g. {"old_key1": "new_key1", "old_key2": "new_key2"}
    Returns:
        a new dictionary with keys changed according to mappings
    """
    if not isinstance(mappings, list):
        mappings = [mappings]
    for mapping in mappings:
       d = {mapping.get(k, k): v for k, v in d.items()}
    return d         

def read_annotation_pickle(path, show_progress=True):
    """
    Returns: A dictionary. Format. scene_id : (bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, image_paths)
    bboxes: numpy array of bounding boxes, shape (N, 9): xyz, lwh, ypr
    object_ids: numpy array of obj ids, shape (N,)
    object_types: list of strings, each string is a type of object
    visible_view_object_dict: a dictionary {view_id: visible_instance_ids}
    extrinsics_c2w: a list of 4x4 matrices, each matrix is the extrinsic matrix of a view
    axis_align_matrix: a 4x4 matrix, the axis-aligned matrix of the scene
    intrinsics: a list of 4x4 matrices, each matrix is the intrinsic matrix of a view
    image_paths: a list of strings, each string is the path of an image in the scene
    """
    with open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
    metainfo = data["metainfo"]
    object_type_to_int = metainfo["categories"]
    object_int_to_type = {v: k for k, v in object_type_to_int.items()}
    datalist = data["data_list"]
    output_data = {}
    pbar = tqdm(range(len(datalist))) if show_progress else range(len(datalist))
    for scene_idx in pbar:
        images = datalist[scene_idx]["images"]
        intrinsic = datalist[scene_idx].get("cam2img", None)  # a 4x4 matrix
        missing_intrinsic = False
        if intrinsic is None:
            missing_intrinsic = True  # each view has different intrinsic for mp3d
        depth_intrinsic = datalist[scene_idx].get(
            "cam2depth", None
        )  # a 4x4 matrix, for 3rscan
        if depth_intrinsic is None and not missing_intrinsic:
            depth_intrinsic = datalist[scene_idx][
                "depth2img"
            ]  # a 4x4 matrix, for scannet
        axis_align_matrix = datalist[scene_idx]["axis_align_matrix"]  # a 4x4 matrix
        scene_id = images[0]["img_path"].split("/")[-2]  # str

        instances = datalist[scene_idx]["instances"]
        bboxes = []
        object_ids = []
        object_types = []
        object_type_ints = []
        for object_idx in range(len(instances)):
            bbox_3d = instances[object_idx]["bbox_3d"]  # list of 9 values
            bbox_label_3d = instances[object_idx]["bbox_label_3d"]  # int
            bbox_id = instances[object_idx]["bbox_id"]  # int
            object_type = object_int_to_type[bbox_label_3d]
            # if object_type in EXCLUDED_OBJECTS:
            #     continue
            object_type_ints.append(bbox_label_3d)
            object_types.append(object_type)
            bboxes.append(bbox_3d)
            object_ids.append(bbox_id)
        bboxes = np.array(bboxes)
        object_ids = np.array(object_ids)
        object_type_ints = np.array(object_type_ints)

        visible_view_object_dict = {}
        extrinsics_c2w = []
        intrinsics = []
        depth_intrinsics = []
        image_paths = []
        for image_idx in range(len(images)):
            img_path = images[image_idx]["img_path"]  # str
            if len(img_path.split("/")) == 3: # should be 4, add prefix
                # example input: posed_images/3rscan0001/000000.jpg
                # example output: 3rscan/posed_images/3rscan0001/000000.jpg
                scene_prefix = get_scene_prefix(img_path)
                img_path = os.path.join(scene_prefix, img_path)
            extrinsic_id = img_path.split("/")[-1].split(".")[0]  # str
            cam2global = images[image_idx]["cam2global"]  # a 4x4 matrix
            if missing_intrinsic:
                intrinsic = images[image_idx]["cam2img"]
                depth_intrinsic = images[image_idx]["cam2depth"]
            visible_instance_indices = images[image_idx][
                "visible_instance_ids"
            ]  # numpy array of int
            visible_instance_ids = object_ids[visible_instance_indices]
            visible_view_object_dict[extrinsic_id] = visible_instance_ids
            extrinsics_c2w.append(cam2global)
            intrinsics.append(intrinsic)
            depth_intrinsics.append(depth_intrinsic)
            image_paths.append(img_path)
        if show_progress:
            pbar.set_description(f"Processing scene {scene_id}")
        output_data[scene_id] = {
            "bboxes": bboxes,
            "object_ids": object_ids,
            "object_types": object_types,
            "object_type_ints": object_type_ints,
            "visible_view_object_dict": visible_view_object_dict,
            "extrinsics_c2w": extrinsics_c2w,
            "axis_align_matrix": axis_align_matrix,
            "intrinsics": intrinsics,
            "depth_intrinsics": depth_intrinsics,
            "image_paths": image_paths,
        }
    return output_data

def read_annotation_pickles(paths):
    """Read multiple annotation pickles and merge them into one dictionary.

    Args:
        paths: a list of paths to annotation pickles.
    Returns: Please refer to the return value of read_annotation_pickle()
    """
    output_data = {}
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        data = read_annotation_pickle(path)
        output_data.update(data)
    output_data = dict(sorted(output_data.items()))
    return output_data

def read_scene_id_mapping(mode):
    assert mode in ["mp3d", "3rscan"]  # scannet do not need this mapping
    fname = f"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/embodiedscan_infos/{mode}_mapping.json"
    if not os.path.exists(fname):
        fname = f"/mnt/petrelfs/linjingli/mmscan_modelzoo-main/embodiedscan_infos/{mode}_mapping.json"
    with open(fname, "r") as f:
        mapping = json.load(f)
    return mapping

RAW2NUM_3RSCAN = read_scene_id_mapping("3rscan")
NUM2RAW_3RSCAN = {v: k for k, v in RAW2NUM_3RSCAN.items()}
RAW2NUM_MP3D = read_scene_id_mapping("mp3d")
NUM2RAW_MP3D = {v: k for k, v in RAW2NUM_MP3D.items()}


def is_valid_name(name):
    is_scannet = "scene" in name or "scannet" in name
    is_3rscan = "3rscan" in name
    is_mp3d = "mp3d" in name or "matterport" in name
    is_valid = is_scannet + is_3rscan + is_mp3d == 1
    if not is_valid:
        print(f"Invalid name {name}")
    return is_valid

def is_sample_idx(name):
    if not is_valid_name(name):
        return False
    length = len(name.split("/"))
    return length >= 2

def is_scene_id(name):
    if not is_valid_name(name):
        return False
    length = len(name.split("/"))
    return length == 1

def sample_idx_to_scene_id(sample_idx):
    """sample index follows the "raw" rule, directly downloaded from the
    internet.

    scene_id follows the "num"bered rule, used in the dataset.
    """
    is_scannet = "scannet" in sample_idx
    is_3rscan = "3rscan" in sample_idx
    is_mp3d = "mp3d" in sample_idx or "matterport" in sample_idx
    assert is_scannet + is_3rscan + is_mp3d == 1, f"Invalid sample_idx {sample_idx}"
    if is_scannet:
        scene_id = sample_idx.split("/")[-1]
    elif is_3rscan:
        raw_id = sample_idx.split("/")[-1]
        scene_id = RAW2NUM_3RSCAN[raw_id]
    elif is_mp3d:
        _, raw_id, region_id = sample_idx.split("/")
        scene_id = RAW2NUM_MP3D[raw_id]
        scene_id = f"{scene_id}_{region_id}"
    return scene_id

def scene_id_to_sample_idx(scene_id):
    is_scannet = "scene" in scene_id
    is_3rscan = "3rscan" in scene_id
    is_mp3d = "mp3d" in scene_id
    assert is_scannet + is_3rscan + is_mp3d == 1, f"Invalid scene_id {scene_id}"
    if is_scannet:
        sample_idx = f"scannet/{scene_id}"
    elif is_3rscan:
        raw_id = NUM2RAW_3RSCAN[scene_id]
        sample_idx = f"3rscan/{raw_id}"
    elif is_mp3d:
        scene_id, region_id = scene_id.split("_region")
        raw_id = NUM2RAW_MP3D[scene_id]
        sample_idx = f"mp3d/{raw_id}/region{region_id}"
    return sample_idx

def to_scene_id(name):
    return name if is_scene_id(name) else sample_idx_to_scene_id(name)

def to_sample_idx(name):
    return name if is_sample_idx(name) else scene_id_to_sample_idx(name)

def read_es_info(path, show_progress=True, count_type_from_zero=False):
    data = np.load(path, allow_pickle=True)
    data_list = data["data_list"]
    object_type_to_int = data["metainfo"]["categories"]
    object_int_to_type = {v: k for k, v in object_type_to_int.items()}
    output_data = {}
    pbar = tqdm(data_list) if show_progress else data_list
    for data in pbar:
        if "sample_idx" in data:
            sample_idx = data["sample_idx"]
            scene_id = sample_idx_to_scene_id(sample_idx)
        else:
            scene_id = data["images"][0]["img_path"].split("/")[-2]  # str
            sample_idx = scene_id_to_sample_idx(scene_id)
        bboxes, object_ids, object_types_int, object_types = [], [], [], []
        for inst in data["instances"]:
            bbox_label_3d = inst["bbox_label_3d"]
            object_type = object_int_to_type[bbox_label_3d]
            bbox_label_3d -= 1 if count_type_from_zero else 0
            bboxes.append(inst["bbox_3d"])
            object_ids.append(inst["bbox_id"])
            object_types_int.append(bbox_label_3d)
            object_types.append(object_type)

        bboxes = np.array(bboxes)
        object_ids = np.array(object_ids)
        object_types_int = np.array(object_types_int)

        output_data[scene_id] = {
            "scene_id": scene_id,
            "sample_idx": sample_idx,
            "bboxes": bboxes,
            "object_ids": object_ids,
            "object_types": object_types,
            "object_type_ints": object_types_int,
        }
    return output_data

def read_es_infos(paths, show_progress=True, count_type_from_zero=False):
    output_data = {}
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        data = read_es_info(path, show_progress, count_type_from_zero)
        output_data.update(data)
    return output_data

if __name__ == "__main__":
    # pickle_file = "D:\Projects\shared_data\embodiedscan_infos\competition_ver\embodiedscan_infos_val.pkl"
    pickle_file = "D:\Projects\shared_data\embodiedscan_infos\embodiedscan_infos_val_full.pkl"
    read_es_infos(pickle_file)
    # read_annotation_pickle(pickle_file)