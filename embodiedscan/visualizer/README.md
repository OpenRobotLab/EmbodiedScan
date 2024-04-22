### EmbodiedScanBaseVisualizer Simple Tutorial

To use visualizer, you need to specify the visualizer in the config. Add the following command to your config file.

```Python
visualizer = dict(type='EmbodiedScanBaseVisualizer', vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
```

Then call the visualizer in models.

```Python
def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
    x = self.extract_feat(batch_inputs_dict, batch_data_samples)
    results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
    predictions = self.add_pred_to_datasample(batch_data_samples, results_list)

    # visualization
    from embodiedscan.visualizer import EmbodiedScanBaseVisualizer
    visualizer = EmbodiedScanBaseVisualizer.get_current_instance()
    visualizer.visualize_scene(predictions)

    return predictions
```

The visualizer will apply Non-Maximum Suppression(NMS) to avoid redundant boxes in the visualization. You can specify its parameters by passing nms_args.

```Python
visualizer.visualize_scene(predictions, nms_args = dict(iou_thr = 0.15, score_thr = 0.075, topk_per_class = 10))
```
