# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from functools import partial
from turtle import forward
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from src.models.convnext import ConvNeXt
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish

class ConvNeXtRGBD(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):
        super().__init__()

        in_chans = 3
        depths = [3, 3, 9, 3]
        # dims = [96, 192, 384, 768]
        dims = [48, 96, 192, 384]
        drop_path_rate = 0.
        layer_scale_init_value = 1e-6
        out_indices = [0, 1, 2, 3]

        self.out_indices = out_indices

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.dims = dims

        se_layers = []
        for i in range(4):
            se_layers.append(SqueezeAndExciteFusionAdd(dims[i], activation=self.activation))
        self.se_layers = nn.Sequential(*se_layers)

        self.rgbEncoder = ConvNeXt(in_chans, depths, dims, drop_path_rate, layer_scale_init_value, out_indices)
        self.depthEncoder = ConvNeXt(in_chans, depths, dims, drop_path_rate, layer_scale_init_value, out_indices)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.dims[0] != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.dims[0],
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.dims[1] != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.dims[1],
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.dims[2] != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.dims[2],
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.dims[3],
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.pretrained = None
        if pretrained_on_imagenet:
            self.pretrained = pretrained_dir # 权重文件路径
            self.init_weights(self.pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.rgbEncoder.apply(_init_weights)
            self.depthEncoder.apply(_init_weights)
            # 写mmcv初始化权重的方法
            # 权重地址 https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth
            pretrain_dict = torch.load('/content/ESANet/convnext_tiny_1k_224.pth')
            rgbEncoder_dict = self.rgbEncoder.state_dict()
            depthEncoder_dict = self.depthEncoder.state_dict()
            # 过滤pretrain中是在model中没有的键值
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in rgbEncoder_dict}
            # 更新model_dict
            rgbEncoder_dict.update(pretrain_dict)
            depthEncoder_dict.update(pretrain_dict)
            self.rgbEncoder.load_state_dict(rgbEncoder_dict)
            self.depthEncoder.load_state_dict(depthEncoder_dict)
            del rgbEncoder_dict
            del depthEncoder_dict
        elif pretrained is None:
            self.rgbEncoder.apply(_init_weights)
            self.depthEncoder.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, rgb, depth):
        if (len(depth.shape) == 5):
            depth = torch.squeeze(depth)
            depth = depth.transpose(1,3).transpose(2,3)
        outs = []
        for i in range(4):
            rgb = self.rgbEncoder.downsample_layers[i](rgb)
            depth = self.depthEncoder.downsample_layers[i](depth)
            rgb = self.rgbEncoder.stages[i](rgb)
            depth = self.depthEncoder.stages[i](depth)
            if i in self.out_indices:
                norm_layer_rgb = getattr(self.rgbEncoder, f'norm{i}')
                norm_layer_depth = getattr(self.depthEncoder, f'norm{i}')
                rgb = norm_layer_rgb(rgb)
                depth = norm_layer_depth(depth)
                rgb = self.se_layers[i](rgb, depth)
                depth = depth
                outs.append(rgb)

        return tuple(outs)

    def forward(self, rgb, depth):
        x = self.forward_features(rgb, depth)
        skip0 = self.skip_layer1(x[0])
        skip1 = self.skip_layer2(x[1])
        skip2 = self.skip_layer3(x[2])
        
        out = self.context_module(x[3])
        out = self.decoder(enc_outs=[out, skip2, skip1, skip0])
        return out


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)

    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return out, out_down_8, out_down_16, out_down_32
        return out


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features

        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x

def main():
    height = 480
    width = 640

    model = ConvNeXtRGBD(
        height=height,
        width=width,
        pretrained_on_imagenet=False)

    print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 3, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)

if __name__ == "__main__":
    main()
    
