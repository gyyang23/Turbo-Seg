# --------------------------------------------------------
# Turbo-Seg
# Written by Guoyu Yang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from turbo_seg.modeling.blocks.convnext_block import LayerNorm, ConvNextBlock, DoubleConv


class Aggregator(nn.Module):
    def __init__(self,
                 text_guidance_dim=640,
                 text_guidance_proj_dim=(128, 64),
                 decoder_dims=(64, 32),
                 decoder_guidance_dims=(512, 256),
                 decoder_guidance_proj_dims=(32, 16),
                 hidden_dim=128,
                 prompt_channel=1,
                 ) -> None:
        """
        Turbo Aggregation Decoder for Turbo-Seg
        Args:
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            hidden_dim: Hidden dimension for convolutional blocks
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.text_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_guidance_dim, tp),
                nn.ReLU(),
            ) for tp in text_guidance_proj_dim
        ])

        self.vis_guidance_projection = nn.ModuleList([
            nn.Sequential(
                LayerNorm(d, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                ConvNextBlock(dp, 5, 0.0, 1.0)
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ])

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])

        self.head = nn.Conv2d(decoder_dims[-1], 1, kernel_size=3, stride=1, padding=1)

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1)  # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1)  # B T P C
        text_feats = text_feats.mean(dim=-2)  # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1)  # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1)  # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1)  # B C H W
        text_feats = F.normalize(text_feats, dim=-1)  # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x, conv):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = conv(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, projected_text_guidance, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, projected_text_guidance[0], guidance[0])
        corr_embed = self.decoder2(corr_embed, projected_text_guidance[1], guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, img_feats, text_feats, vis_feats):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C)
            apperance_guidance: tuple of (B, C, H, W)
        """

        corr = self.correlation(img_feats, text_feats)
        corr_embed = self.corr_embed(corr, self.conv1)

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = [proj(text_feats) for proj in self.text_guidance_projection]

        if self.vis_guidance_projection is not None:
            projected_vis_guidance = [proj(v) for proj, v in zip(self.vis_guidance_projection, vis_feats[1:])]

        logit = self.conv_decoder(corr_embed, projected_text_guidance, projected_vis_guidance)

        return logit


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.spatial_aggregator = ConvNextBlock(in_channels, 5, 0.0, 1.0)
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.text_fusion = nn.Sequential(
            DoubleConv(2 * in_channels, in_channels)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.vis_fusion = nn.Sequential(
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x, text_guidance, vis_guidance):
        B, _, _, _ = vis_guidance.size()
        x = self.spatial_aggregator(x)

        _, _, H, W = x.size()
        text_guidance = repeat(text_guidance, "B T C -> (B T) C H W", H=H, W=W)
        x = torch.cat([self.norm(x), text_guidance], dim=1)
        x = self.text_fusion(x)

        x = self.up(x)
        T = x.size(0) // vis_guidance.size(0)
        guidance = repeat(vis_guidance, "B C H W -> (B T) C H W", T=T)
        x = torch.cat([x, guidance], dim=1)
        x = self.vis_fusion(x)

        return x

