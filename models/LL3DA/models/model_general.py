import importlib

import torch
from torch import nn


class CaptionNet(nn.Module):

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector is True:
            self.detector.eval()
            for param in self.detector.parameters():
                param.requires_grad = False
        return self

    def pretrained_parameters(self):
        if hasattr(self.captioner, 'pretrained_parameters'):
            return self.captioner.pretrained_parameters()
        else:
            return []

    def __init__(self, args, dataset_config, train_dataset):
        super(CaptionNet, self).__init__()

        self.freeze_detector = args.freeze_detector
        self.detector = None
        self.captioner = None

        if args.detector is not None:
            detector_module = importlib.import_module(
                f'models.{args.detector}.detector')
            self.detector = detector_module.detector(args, dataset_config)

        if args.captioner is not None:
            captioner_module = importlib.import_module(
                f'models.{args.captioner}.captioner')
            self.captioner = captioner_module.captioner(args, train_dataset)

        self.train()

    def forward(self,
                batch_data_label: dict,
                is_eval: bool = False,
                task_name: str = None) -> dict:

        outputs = {'loss': torch.zeros(1)[0].cuda()}

        # in the LL3DA paper, the detector is always freeze.
        if self.detector is not None:
            if self.freeze_detector is True:
                outputs = self.detector(batch_data_label, is_eval=True)
            else:
                outputs = self.detector(batch_data_label, is_eval=is_eval)

        # so it's no need to count loss
        if self.freeze_detector is True:
            outputs['loss'] = torch.zeros(1)[0].cuda()

        # this is the output of the detector
        # box_predictions['outputs'].update({
        #     'prop_features': box_features.permute(0, 2, 1, 3),  # nlayers x batch x nqueries x channel   # the feature of each proposal (box)
        #     'enc_features': enc_features.permute(1, 0, 2),      # batch x npoints x channel   # the feature of the whole scene
        #     'enc_xyz': enc_xyz,      # batch x npoints x 3
        #     'query_xyz': query_xyz,  # batch x nqueries x 3
        # })
        # "sem_cls_logits": cls_logits[l],
        #         "center_normalized": center_normalized.contiguous(),
        #         "center_unnormalized": center_unnormalized,
        #         "size_normalized": size_normalized[l],
        #         "size_unnormalized": size_unnormalized,
        #         "angle_logits": angle_logits[l],
        #         "angle_residual": angle_residual[l],
        #         "angle_residual_normalized": angle_residual_normalized[l],
        #         "angle_continuous": angle_continuous,
        #         "objectness_prob": objectness_prob,
        #         "sem_cls_prob": semcls_prob,
        #         "box_corners": box_corners,

        if self.captioner is not None:
            outputs = self.captioner(outputs,
                                     batch_data_label,
                                     is_eval=is_eval,
                                     task_name=task_name)
        else:
            batch, nproposals, _, _ = outputs['box_corners'].shape
            outputs['lang_cap'] = [['this is a valid match!'] * nproposals
                                   ] * batch
        return outputs
