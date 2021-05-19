from segmentation_models_pytorch.base import initialization as init

import torch.nn as nn
import torch
from segmentation_models_pytorch.base.modules import Flatten, Activation

from typing import Optional, Union, List
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class UnetVFLOW(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
        )

        self.xydir_head = EncoderRegressionHead(
            in_channels=self.encoder.out_channels[-1],
            out_channels=2,
        )

        self.height_head = RegressionHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )

        self.mag_head = RegressionHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )

        self.scale_head = ScaleHead()

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.xydir_head)
        init.initialize_head(self.height_head)
        init.initialize_head(self.mag_head)
        init.initialize_head(self.scale_head)

    def forward(self, x):

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        xydir = self.xydir_head(features[-1])
        height = self.height_head(decoder_output)
        mag = self.mag_head(decoder_output)
        scale = self.scale_head(mag, height)

        if scale.ndim == 0:
            scale = torch.unsqueeze(scale, axis=0)

        return xydir, height, mag, scale

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        identity = nn.Identity()
        activation = Activation(None)
        super().__init__(conv2d, identity, activation)


class EncoderRegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=0.5, inplace=True)
        linear = nn.Linear(in_channels, 2, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class ScaleHead(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.flatten = torch.flatten
        self.dot = torch.dot

    def forward(self, mag, height):
        curr_mag = self.flatten(mag, start_dim=1)
        curr_height = self.flatten(height, start_dim=1)
        batch_size = curr_mag.shape[0]
        length = curr_mag.shape[1]
        denom = (
            torch.squeeze(
                torch.bmm(
                    curr_height.view(batch_size, 1, length),
                    curr_height.view(batch_size, length, 1),
                )
            )
            + 0.01
        )
        pinv = curr_height / denom.view(batch_size, 1)
        scale = torch.squeeze(
            torch.bmm(
                pinv.view(batch_size, 1, length), curr_mag.view(batch_size, length, 1)
            )
        )
        return scale
