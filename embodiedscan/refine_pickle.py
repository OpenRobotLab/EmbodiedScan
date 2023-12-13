import json
import os
import pickle

from tqdm import tqdm


def path_split(path):
    s = path.split('/')
    return s[0], s[2], s[3]


with open(
        '/mnt/petrelfs/share_data/maoxiaohan/3rscan/meta_data/' +
        '3rscan_mapping.json', 'r') as f:
    map_3rscan = json.load(f)
back_3rscan = {v: k for k, v in map_3rscan.items()}

with open(
        '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/meta_data/' +
        'scene_mapping.json', 'r') as f:
    map_mp3d = json.load(f)
back_mp3d = {v: k for k, v in map_mp3d.items()}
buildings = os.listdir(
    '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/rename')
assert len(buildings) == len(list(back_mp3d.keys()))
max_cam = 0
back_mp3d_cam = dict()
for building in buildings:
    assert building[-5:] == '.json'
    building_name = building[:-5]
    with open(
            os.path.join(
                '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/rename',
                building), 'r') as f:
        tmp = json.load(f)
    max_cam = max(max_cam, len(list(tmp.keys())))
    back_mp3d_cam[building_name] = {v: k for k, v in tmp.items()}

print(max_cam)


def mp3d_split(region, camera):
    global back_mp3d
    global back_mp3d_cam
    x = region.find('_region')
    building = region[:x]
    raw_building = back_mp3d[building]
    raw_region = region[x + 1:]
    assert camera[-4] == '_'
    raw_camera = back_mp3d_cam[raw_building][camera[:-4]]
    cam_pos = camera[-3:]
    return raw_building, raw_region, raw_camera, cam_pos


def generate(in_dir, out_dir, filename):
    with open(os.path.join(in_dir, filename), 'rb') as f:
        data = pickle.load(f)

    for scene in tqdm(data['data_list']):
        bo = False
        for img in scene['images']:
            path = img['img_path']
            dataset, region, camera = path_split(path)
            assert camera[-4:] == '.jpg'
            camera = camera[:-4]

            if dataset == 'scannet':
                img_path = path
                depth_path = f'{dataset}/posed_images/{region}/{camera}.png'
                img['depth_path'] = depth_path
                if not bo:
                    scene['depth_cam2img'] = scene['depth2img']
                    scene.pop('depth2img', None)
                    scene['sample_idx'] = f'scannet/{region}'
                    bo = True
            elif dataset == '3rscan':
                raw_region = back_3rscan[region]
                img_path = f'{dataset}/{raw_region}/sequence/' + \
                    'frame-{camera}.color.jpg'
                depth_path = f'{dataset}/{raw_region}/sequence/' + \
                    'frame-{camera}.depth.pgm'
                img['img_path'] = img_path
                img['depth_path'] = depth_path
                if not bo:
                    scene['depth_cam2img'] = scene['cam2depth']
                    scene.pop('cam2depth', None)
                    scene['sample_idx'] = f'3rscan/{raw_region}'
                    bo = True
            elif dataset == 'matterport3d':
                raw_building, raw_region, raw_camera, cam_pos = mp3d_split(
                    region, camera)
                img_path = f'{dataset}/{raw_building}/' + \
                    'matterport_color_images/{raw_camera}_i{cam_pos}.jpg'
                depth_path = f'{dataset}/{raw_building}/' + \
                    'matterport_depth_images/{raw_camera}_d{cam_pos}.png'
                img['img_path'] = img_path
                img['depth_path'] = depth_path
                img.pop('cam2depth', None)
                if not bo:
                    scene['sample_idx'] = \
                        f'matterport3d/{raw_building}/{raw_region}'
                    bo = True
            else:
                raise NotImplementedError

    with open(os.path.join(out_dir, filename), 'wb') as f:
        pickle.dump(data, f)


generate(in_dir='/mnt/petrelfs/share_data/wangtai/data/full_10_visible',
         out_dir='./data',
         filename='embodiedscan_infos_train_full.pkl')
generate(in_dir='/mnt/petrelfs/share_data/wangtai/data/full_10_visible',
         out_dir='./data',
         filename='embodiedscan_infos_val_full.pkl')
