import open_clip
import torch
from torch import nn
import json

from turbo_seg import imagenet_templates


class CLIP(nn.Module):
    def __init__(
            self,
            model_name: str = "Convnext-B",
            prompt_ensemble_type: str = "single",
            train_class_json: str = "datasets/coco.json",
            test_class_json: str = "datasets/coco.json",
            device: str = "cuda",
    ):
        """
        CLIP backbone for visual-language modeling
        Args:
            model_name: CLIP model name (e.g., "Convnext-B" or "Convnext-L")
            prompt_ensemble_type: type of prompt ensemble
            device: device to run on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.prompt_ensemble_type = prompt_ensemble_type

        """Initialize CLIP model based on configuration"""
        self.tokenizer = None
        if self.model_name == "Convnext-B":
            name, pretrain = ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k')
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device=self.device)
            self.clip_model = self.clip_model.float()
            self.tokenizer = open_clip.get_tokenizer(name)
        elif self.model_name == "Convnext-L":
            name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device=self.device)
            self.clip_model = self.clip_model.float()
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            raise NotImplementedError(f"Unknown model: {self.model_name}")

        """Initialize prompt templates based on ensemble type"""
        if self.prompt_ensemble_type == "imagenet_select":
            self.prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            self.prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            self.prompt_templates = ['A photo of a {} in the scene', ]
        else:
            raise NotImplementedError

        """Initialize text_features"""
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts is not None
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts

        self.tokens = None
        self.cache = None

    def _get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache

        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]

                texts = self.tokenizer(texts).cuda()
                tokens.append(texts)

            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unsqueeze(1)

        if not self.training:
            self.cache = class_embeddings

        return class_embeddings
