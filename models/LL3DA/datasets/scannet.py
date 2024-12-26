from datasets.scannet_base_dataset import (BASE, DatasetConfig,
                                           ScanNetBaseDataset)
from eval_utils.evaluate_det import evaluate


class Dataset(ScanNetBaseDataset):

    def __init__(
        self,
        args,
        dataset_config,
        split_set='train',
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )

        self.eval_func = evaluate
