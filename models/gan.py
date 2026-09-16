"""Adversarial (generative) super-resolution components.

``ESRGANDiscriminator`` is the ESRGAN-style VGG discriminator adapted for an
arbitrary band count (works for 3-band RGB or 4-band RGBNIR). It uses an
adaptive pooling tail so it accepts any input spatial size.
"""

import torch
import torch.nn as nn


class ESRGANDiscriminator(nn.Module):
    """VGG-style discriminator with a scalar logit output (use BCEWithLogits)."""

    def __init__(self, in_channels=4, num_feat=64):
        super().__init__()

        def conv_block(n_in, n_out, stride=1, norm=True):
            layers = [nn.Conv2d(n_in, n_out, 3, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(n_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.body = nn.Sequential(
            conv_block(in_channels, num_feat, 1, norm=False),
            conv_block(num_feat, num_feat, 2),
            conv_block(num_feat, num_feat * 2, 1),
            conv_block(num_feat * 2, num_feat * 2, 2),
            conv_block(num_feat * 2, num_feat * 4, 1),
            conv_block(num_feat * 4, num_feat * 4, 2),
            conv_block(num_feat * 4, num_feat * 8, 1),
            conv_block(num_feat * 8, num_feat * 8, 2),
            conv_block(num_feat * 8, num_feat * 8, 1),
            conv_block(num_feat * 8, num_feat * 8, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_feat * 8, 1),
        )

    def forward(self, x):
        return self.classifier(self.body(x))