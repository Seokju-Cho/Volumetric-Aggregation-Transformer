import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.swin import SwinTransformer2d, TransformerWarper2d
from .base.our_conv4d import Interpolate4d, Encoder4D
from .base.swin4d import SwinTransformer, TransformerWarper


class OurModel(nn.Module):
    def __init__(self,
        inch=(3, 23, 4),
    ):
        super().__init__()

        self.encoders = nn.ModuleList([
            Encoder4D( # Encoder for conv_5
                corr_levels=(inch[0], 64, 128),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 1, 1),
                    (1, 1, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                group=(4, 8),
                residual=False
            ),
            Encoder4D( # Encoder for conv_4
                corr_levels=(inch[1], 64, 128),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                    (1, 1, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                group=(4, 8),
                residual=False
            ),
            Encoder4D( # Encoder for conv_3
                corr_levels=(inch[2], 32, 64, 128),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                    (1, 1, 2, 2),
                    (1, 1, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                group=(2, 4, 8,),
                residual=False
            ),
        ])

        self.transformer = nn.ModuleList([
            TransformerWarper(SwinTransformer(
                corr_size=(8, 8, 8, 8),
                embed_dim=128,
                depth=4,
                num_head=1,
                window_size=4,
            )),
            TransformerWarper(SwinTransformer(
                corr_size=(16, 16, 8, 8),
                embed_dim=128,
                depth=2,
                num_head=1,
                window_size=4,
            )),
            TransformerWarper(SwinTransformer(
                corr_size=(32, 32, 8, 8),
                embed_dim=128,
                depth=2,
                num_head=1,
                window_size=4,
            )),
        ])

        self.upscale = nn.ModuleList([
            Interpolate4d(size=(16, 16), dim='query'),
            Interpolate4d(size=(32, 32), dim='query'),
            Interpolate4d(size=(64, 64), dim='query'),
        ])

        self.swin_decoder = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(32, 32), embed_dim=128 + 64, window_size=8, num_heads=[1])
                ),
                nn.Conv2d(128 + 64, 96, 1)
            ),
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(32, 32), embed_dim=96 + 32, window_size=8, num_heads=[1])
                ),
                nn.Conv2d(96 + 32, 48, 1)
            ),
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(32, 32), embed_dim=48 + 16, window_size=8, num_heads=[1])
                ),
            )
        ])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 2, (3, 3), padding=(1, 1), bias=True)
        )

        self.dropout2d = nn.Dropout2d(p=0.5)

        self.proj_query_feat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 64, 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 32, 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 16, 1),
                nn.ReLU(),
            )
        ])
    
    def extract_last(self, x):
        return [k[:, -1] for k in x]

    def forward(self, hypercorr_pyramid, query_feats, support_mask):
        query_feat5, query_feat4, query_feat3, query_feat2 = self.extract_last(query_feats)
        query_feat5, query_feat4, query_feat3 = [
            self.proj_query_feat[i](x) for i, x in enumerate((query_feat5, query_feat4, query_feat3)) 
        ]

        corr5 = self.encoders[0](hypercorr_pyramid[0])[0]
        corr4 = self.encoders[1](hypercorr_pyramid[1])[0]
        corr3 = self.encoders[2](hypercorr_pyramid[2])[0]
        
        corr5 = corr5 + self.transformer[0](corr5)
        corr5_upsampled = self.upscale[0](corr5)

        corr4 += corr5_upsampled
        corr4 = corr4 + self.transformer[1](corr4)
        corr4_upsampled = self.upscale[1](corr4)

        corr3 += corr4_upsampled
        corr3 = corr3 + self.transformer[2](corr3)
        x = corr3.mean(dim=(-2, -1))

        query_feat5 = F.interpolate(query_feat5, size=(32, 32), mode='bilinear', align_corners=True)
        x = self.swin_decoder[0](torch.cat((x, query_feat5), dim=1))
        query_feat4 = F.interpolate(query_feat4, size=(32, 32), mode='bilinear', align_corners=True)
        x = self.swin_decoder[1](torch.cat((x, query_feat4), dim=1))
        query_feat3 = F.interpolate(query_feat3, size=(32, 32), mode='bilinear', align_corners=True)
        x = self.swin_decoder[2](torch.cat((x, query_feat3), dim=1))

        return self.decoder(x)