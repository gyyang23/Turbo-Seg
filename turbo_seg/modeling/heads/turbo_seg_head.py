import torch
from detectron2.layers import ShapeSpec
from torch import nn
from typing import Dict, List, Tuple
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import configurable

from .aggregator import Aggregator


@SEM_SEG_HEADS_REGISTRY.register()
class TurboSegHead(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            num_classes: int,
            ignore_value: int,
            text_guidance_dim: int,
            text_guidance_proj_dim: int,
            decoder_dims: List[int],
            decoder_guidance_dims: List[int],
            decoder_guidance_proj_dims: List[int],
            hidden_dims: Tuple[int],
            num_prompts: int,
    ):
        """
        TurboSeg segmentation head
        Args:
            num_classes: Number of output classes
            ignore_value: Ignore value for loss calculation
            text_guidance_dim: Dimension for text guidance
            text_guidance_proj_dim: Projection dimension for text guidance
            decoder_dims: Dimensions for decoder layers
            decoder_guidance_dims: Guidance dimensions for decoder
            decoder_guidance_proj_dims: Projection dimensions for decoder guidance
            hidden_dims: Hidden dimensions for Turbo Aggregation Decoder
            num_prompts: Number of prompt templates used
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_value = ignore_value

        self.aggregator = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            hidden_dim=hidden_dims,
            prompt_channel=num_prompts,
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "text_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
            "text_guidance_proj_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM,
            "decoder_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS,
            "decoder_guidance_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS,
            "decoder_guidance_proj_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS,
            "hidden_dims": cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS,
            "num_prompts": cfg.MODEL.SEM_SEG_HEAD.NUM_PROMPTS,
        }

    def forward(self, visual_features: Dict[str, torch.Tensor],
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            visual_features: Dictionary of visual features from backbone
            text_embeddings: Text embeddings from CLIP backbone
        Returns:
            Segmentation logits
        """
        # Process visual features
        vis_features = [visual_features[k] for k in visual_features.keys()][::-1]

        # Repeat text embeddings for batch dimension
        text_embeddings = text_embeddings.repeat(vis_features[0].shape[0], 1, 1, 1)

        # Run through aggregator
        return self.aggregator(vis_features[0], text_embeddings, vis_features)