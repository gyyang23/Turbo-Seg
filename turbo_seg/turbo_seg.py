# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from turbo_seg.modeling.backbone.clip import CLIP


@META_ARCH_REGISTRY.register()
class TurboSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        train_class_json: str,
        test_class_json: str,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        clip_finetune: str,
        clip_resolution: Tuple[int],
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = CLIP(
            model_name=clip_pretrained,
            prompt_ensemble_type=prompt_ensemble_type,
            train_class_json=train_class_json,
            test_class_json=test_class_json
        )
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.clip_finetune = clip_finetune
        for name, params in self.backbone.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "conv":
                    params.requires_grad = True if "conv" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                elif clip_finetune == "mlp":
                    params.requires_grad = True if "mlp" in name or "position" in name else False
                elif clip_finetune == "full_res5":
                    if "stages.3" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
            elif "transformer" in name:
                if clip_finetune == "full":
                    params.requires_grad = True
            else:
                params.requires_grad = False

        self.gt_classes = None
        self.prompt = None
        self.clip_resolution = clip_resolution

    @classmethod
    def from_config(cls, cfg):
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "clip_pretrained": cfg.MODEL.CLIP_PRETRAINED,
            "prompt_ensemble_type": cfg.MODEL.PROMPT_ENSEMBLE_TYPE,
            "train_class_json": cfg.MODEL.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.TEST_CLASS_JSON,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "clip_finetune": cfg.MODEL.CLIP_FINETUNE,
            "clip_resolution": cfg.INPUT.CROP.SIZE,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList(
            tensor=torch.stack(clip_images),
            image_sizes=[img.shape[-2:] for img in clip_images]
        )

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.backbone.clip_model.encode_image(clip_images_resized, dense=True)

        res3 = clip_features['res3']
        res4 = clip_features['res4']
        clip_vis_dense = clip_features['clip_vis_dense']
        features = {'res3': res3, 'res4': res4, 'clip_vis_dense': clip_vis_dense}

        text = self.backbone.class_texts if self.training else self.backbone.test_class_texts
        text = [text[c] for c in self.gt_classes] if self.gt_classes is not None else text
        text = self.backbone._get_text_embeds(text, self.backbone.prompt_templates, self.backbone.clip_model, self.prompt)

        outputs = self.sem_seg_head(features, text)

        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)

            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0, 2, 3, 1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot

            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results
