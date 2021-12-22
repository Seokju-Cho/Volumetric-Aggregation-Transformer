import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.swin import SwinTransformer2d, TransformerWarper2d
from model.base.our_conv4d import Interpolate4d, Encoder4D
from model.base.swin4d import SwinTransformer, TransformerWarper


class OurModel(nn.Module):
    def __init__(
        self,
        inch=(3, 23, 4),
        feature_affinity=(True, True, True),
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
                num_head=4,
                window_size=4,
            )),
            TransformerWarper(SwinTransformer(
                corr_size=(16, 16, 8, 8),
                embed_dim=128,
                depth=2,
                num_head=4,
                window_size=4,
            )),
            TransformerWarper(SwinTransformer(
                corr_size=(32, 32, 8, 8),
                embed_dim=128,
                depth=2,
                num_head=4,
                window_size=4,
            )),
        ])

        self.upscale = nn.ModuleList([
            Interpolate4d(size=(16, 16), dim='query'),
            Interpolate4d(size=(32, 32), dim='query'),
            Interpolate4d(size=(64, 64), dim='query'),
        ])

        self.feature_affinity = feature_affinity
        decoder_dim = [
            (128 + 64) if feature_affinity[0] else 128,
            (96 + 32) if feature_affinity[1] else 96,
            (48 + 16) if feature_affinity[2] else 48
        ]

        self.swin_decoder = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(32, 32), embed_dim=decoder_dim[0], window_size=8)
                ),
                nn.Conv2d(decoder_dim[0], 96, 1)
            ),
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(64, 64), embed_dim=decoder_dim[1], window_size=8)
                ),
                nn.Conv2d(decoder_dim[1], 48, 1)
            ),
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(128, 128), embed_dim=decoder_dim[2], window_size=8)
                ),
            )
        ])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_dim[2], 32, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 2, (3, 3), padding=(1, 1), bias=True)
        )

        self.dropout2d = nn.Dropout2d(p=0.5)

        self.proj_query_feat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 64, 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 32, 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 16, 1),
                nn.ReLU(),
            )
        ])

    
    def extract_last(self, x):
        return [k[:, -1] for k in x]

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode='nearest') for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [F.interpolate(x, size=size, mode='nearest') for x, size in zip(feats, sizes)]
        return recoverd_feats

    def forward(self, hypercorr_pyramid, query_feats, support_mask):
        _, query_feat4, query_feat3, query_feat2 = self.extract_last(query_feats)
        query_feat4, query_feat3, query_feat2 = [
            self.proj_query_feat[i](x) for i, x in enumerate((query_feat4, query_feat3, query_feat2)) 
        ]

        query_feat4, query_feat3 = self.apply_dropout(self.dropout2d, query_feat4, query_feat3)

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

        x = self.swin_decoder[0](torch.cat((x, query_feat4), dim=1) if self.feature_affinity[0] else x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        x = self.swin_decoder[1](torch.cat((x, query_feat3), dim=1) if self.feature_affinity[1] else x)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        x = self.swin_decoder[2](torch.cat((x, query_feat2), dim=1) if self.feature_affinity[2] else x)

        return self.decoder(x)
