import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


def make_dir(dir_path):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_imgs(img_paths, option=cv2.IMREAD_COLOR):
    imgs = [cv2.imread(img_path, option) for img_path in img_paths]
    return imgs


def load_pickle(filename):
    with Path(filename).open('rb') as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with Path(filename).open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with Path(filename).open('rb') as f:
        return json.load(f)


def save_json(data, filename, save_pretty=True, sort_keys=False):
    with Path(filename).open('w') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with Path(filename).open('r') as f:
        return [json.loads(l.strip('\n')) for l in f.readlines()]


def save_jsonl(data, filename):
    with Path(filename).open('w') as f:
        f.write('\n'.join([json.dumps(e) for e in data]))


def load_yaml(filename):
    with Path(filename).open('r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def save_yaml(data, filename):
    with Path(filename).open('w') as f:
        json.dump(OmegaConf.to_container(data, resolve=True), f, indent=2)


def load_csv(filename, delimiter=','):
    idx2key = None
    contents = {}
    with Path(filename).open('r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for l_idx, row in reader:
            if l_idx == 0:
                idx2key = row
                for k_idx, key in enumerate(idx2key):
                    contents[key] = []
            else:
                for c_idx, col in enumerate(row):
                    contents[idx2key[c_idx]].append(col)
    return contents, idx2key


def save_csv(data, filename, cols=None, delimiter=','):
    with Path(filename).open('w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        num_entries = len(data[list(data.keys())[0]])
        assert cols is not None, 'Must have column names for dumping csv files.'
        writer.writerow(cols)
        for l_idx in range(num_entries):
            row = [data[key][l_idx] for key in cols]
            writer.writerow(row)


def load_numpy(filename):
    return np.load(filename, allow_pickle=True)


def save_numpy(data, filename):
    np.save(filename, data, allow_pickle=True)


def load_tensor(filename):
    return torch.load(filename)


def save_tensor(data, filename):
    torch.save(data, filename)
