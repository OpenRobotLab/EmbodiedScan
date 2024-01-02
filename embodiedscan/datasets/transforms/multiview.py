import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose

from embodiedscan.registry import TRANSFORMS
from embodiedscan.structures.points import get_points_type


@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):

    def __init__(self, transforms, n_images, ordered=False):
        super().__init__()
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.ordered = ordered

    def transform(self, results: dict) -> dict:
        imgs = []
        points = []
        intrinsics = []
        extrinsics = []
        ids = np.arange(len(results['img_path']))
        replace = True if self.n_images > len(ids) else False
        if self.ordered:
            step = (len(ids) - 1) // (self.n_images - 1
                                      )  # TODO: BUG, fix from branch fbocc
            if step > 0:
                ids = ids[::step]
                # sometimes can not get the accurate n_images in this way
                # then take the first n_images one
                ids = ids[:self.n_images]
            # else: all the ids are evaluated
        else:
            ids = np.random.choice(ids, self.n_images, replace=replace)
        for i in ids.tolist():
            _results = dict()
            _results['img_path'] = results['img_path'][i]
            if 'depth_img_path' in results:
                _results['depth_img_path'] = results['depth_img_path'][i]
                if isinstance(results['depth_cam2img'], list):
                    _results['depth_cam2img'] = results['depth_cam2img'][i]
                    _results['cam2img'] = results['depth2img']['intrinsic'][i]
                else:
                    _results['depth_cam2img'] = results['depth_cam2img']
                    _results['cam2img'] = results['cam2img']
                _results['depth_shift'] = results['depth_shift']
            _results = self.transforms(_results)
            if 'depth_shift' in _results:
                _results.pop('depth_shift')
            if 'img' in _results:
                imgs.append(_results['img'])
            if 'points' in _results:
                points.append(_results['points'])
            if isinstance(results['depth2img']['intrinsic'], list):
                intrinsics.append(results['depth2img']['intrinsic'][i])
            else:
                intrinsics.append(results['depth2img']['intrinsic'])
            extrinsics.append(results['depth2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'points']:
                results[key] = _results[key]
        if len(imgs):
            results['img'] = imgs
        if len(points):
            results['points'] = points
        if 'visible_instance_masks' in results:
            results['visible_instance_masks'] = [
                results['visible_instance_masks'][i] for i in ids
            ]
        if 'visible_occupancy_masks' in results:
            results['visible_occupancy_masks'] = [
                results['visible_occupancy_masks'][i] for i in ids
            ]
        results['depth2img']['intrinsic'] = intrinsics
        results['depth2img']['extrinsic'] = extrinsics

        return results


@TRANSFORMS.register_module()
class AggregateMultiViewPoints(BaseTransform):

    def __init__(self,
                 coord_type: str = 'DEPTH',
                 save_slices: bool = False) -> None:
        super().__init__()
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.save_slices = save_slices

    def transform(self, results: dict) -> dict:
        points = results[
            'points']  # TODO: transforms should use numpy instead of torch
        global_points = []
        points_slice_indices = [0]
        for idx in range(len(points)):
            point = points[idx].tensor[..., :3]
            point = torch.cat([point, point.new_ones(point.shape[0], 1)],
                              dim=1)
            global2ego = torch.from_numpy(
                results['depth2img']['extrinsic'][idx]).to(point.device)
            global_point = (torch.linalg.solve(global2ego, point.transpose(
                0, 1))).transpose(0, 1)
            points[idx].tensor[:, :3] = global_point[:, :3]
            global_points.append(points[idx].tensor)
            if self.save_slices:
                points_slice_indices.append(points_slice_indices[-1] +
                                            len(points[idx].tensor))
        points = torch.cat(global_points)
        # a little hard code, to be improved
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points,
            points_dim=results['points'][0].points_dim,
            attribute_dims=results['points'][0].attribute_dims)
        results['points'] = points

        if self.save_slices:
            results['points_slice_indices'] = points_slice_indices

        # TODO: fix the packing of results['img'] and remove this code
        # del results['img']

        return results


@TRANSFORMS.register_module()
class ConstructMultiSweeps(BaseTransform):

    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        points = results['points']
        points_slice_indices = results['points_slice_indices']
        points_slice_indices = results['points_slice_indices']
        cumulated_points = points.tensor[
            points_slice_indices[0]:points_slice_indices[1]]
        batch_points = [cumulated_points]

        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        batch_gt_bboxes_3d = gt_bboxes_3d
        batch_gt_labels_3d = gt_labels_3d

        if 'visible_instance_masks' in results:
            visible_instance_masks = results['visible_instance_masks']
            visible_instance_ids = []
            for idx in range(len(visible_instance_masks)):
                visible_instance_ids.append(
                    set(
                        np.argwhere(np.array(
                            visible_instance_masks[idx])).flatten()))
            cumulated_ids = set(visible_instance_ids[0])
            indices = np.array(list(cumulated_ids), dtype=np.int32)
            batch_gt_bboxes_3d = [gt_bboxes_3d[indices]]
            batch_gt_labels_3d = [gt_labels_3d[indices]]

        if 'visible_occupancy_masks' in results:
            visible_occupancy_masks = results['visible_occupancy_masks']
            cumulated_masks = visible_occupancy_masks[0]
            batch_gt_occupancy_masks = [visible_occupancy_masks[0]]

        for idx in range(1, len(points_slice_indices) - 1):
            # construct sparse tensor and features
            start = points_slice_indices[idx]
            end = points_slice_indices[idx + 1]
            cumulated_points = torch.cat(
                [cumulated_points, points.tensor[start:end]])
            batch_points.append(cumulated_points)

            if 'visible_instance_masks' in results:
                cumulated_ids = cumulated_ids.union(visible_instance_ids[idx])
                indices = np.array(list(cumulated_ids), dtype=np.int32)
                batch_gt_bboxes_3d.append(gt_bboxes_3d[indices])
                batch_gt_labels_3d.append(gt_labels_3d[indices])

            if 'visible_occupancy_masks' in results:
                cumulated_masks = np.logical_or(cumulated_masks,
                                                visible_occupancy_masks[idx])
                batch_gt_occupancy_masks.append(cumulated_masks)

        results['points'] = batch_points

        if 'visible_instance_masks' in results:
            results['gt_bboxes_3d'] = batch_gt_bboxes_3d
            results['gt_labels_3d'] = batch_gt_labels_3d
            if 'eval_ann_info' in results:
                results['eval_ann_info']['gt_bboxes_3d'] = results[
                    'gt_bboxes_3d']
                results['eval_ann_info']['gt_labels_3d'] = results[
                    'gt_labels_3d']

        if 'visible_occupancy_masks' in results:
            results['gt_occupancy_masks'] = batch_gt_occupancy_masks
            if 'eval_ann_info' in results:
                results['eval_ann_info']['gt_occupancy_masks'] = results[
                    'gt_occupancy_masks']

        return results
