import os

import open3d as o3d
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from embodiedscan.registry import VISUALIZERS
from embodiedscan.visualization.utils import _9dof_to_box, nms_filter


@VISUALIZERS.register_module()
class EmbodiedBaseVisualizer(Visualizer):

    def __init__(self,
                 name: str = 'visualizer',
                 save_dir: str = None,
                 vis_backends=None) -> None:
        super().__init__(name=name,
                         vis_backends=vis_backends,
                         save_dir=save_dir)

    @staticmethod
    def get_root_dir(img_path):
        if 'posed_images' in img_path:
            return img_path.split('posed_images')[0]
        if 'sequence' in img_path:
            return img_path.split('sequence')[0]
        if 'matterport_color_images' in img_path:
            return img_path.split('matterport_color_images')[0]
        raise ValueError('Custom datasets are not supported.')

    @staticmethod
    def get_ply(root_dir, scene_name):
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s
        if dataset == 'scannet':
            filepath = os.path.join(root_dir, 'scans', region,
                                    f'{region}_vh_clean.ply')
        elif dataset == '3rscan':
            filepath = os.path.join(root_dir, 'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            filepath = os.path.join(root_dir, 'region_segmentations',
                                    f'{region}.ply')
        else:
            raise NotImplementedError
        return filepath

    @master_only
    def visual_scene(self, data_samples, class_filter=None, nms_args=dict()):
        assert len(data_samples) == 1
        data_sample = data_samples[0]

        metainfo = data_sample.metainfo
        pred = data_sample.pred_instances_3d
        gt = data_sample.eval_ann_info

        if not hasattr(pred, 'labels_3d'):
            assert gt['gt_labels_3d'].shape[0] == 1
            gt_label = gt['gt_labels_3d'][0].item()
            _ = pred.bboxes_3d.tensor.shape[0]
            pseudo_label = pred.bboxes_3d.tensor.new_ones(_, ) * gt_label
            pred.labels_3d = pseudo_label
        pred_box, pred_label = nms_filter(pred, **nms_args)

        root_dir = self.get_root_dir(metainfo['img_path'][0])
        ply_file = self.get_ply(root_dir, metainfo['scan_id'])
        axis_align_matrix = metainfo['axis_align_matrix']

        mesh = o3d.io.read_triangle_mesh(ply_file, True)
        mesh.transform(axis_align_matrix)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        boxes = []
        # pred 3D box
        n = pred_box.shape[0]
        for i in range(n):
            box = pred_box[i]
            label = pred_label[i]
            if class_filter is not None and label != class_filter:
                continue
            box_geo = _9dof_to_box(box, color=(255, 0, 0))
            boxes.append(box_geo)
        # gt 3D box
        m = gt['gt_bboxes_3d'].tensor.shape[0]
        for i in range(m):
            box = gt['gt_bboxes_3d'].tensor[i]
            label = gt['gt_labels_3d'][i]
            if class_filter is not None and label != class_filter:
                continue
            box_geo = _9dof_to_box(box, color=(0, 255, 0))
            boxes.append(box_geo)

        o3d.visualization.draw_geometries([mesh, frame] + boxes)
