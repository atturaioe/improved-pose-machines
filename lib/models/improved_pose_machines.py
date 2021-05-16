import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.googlenet import GoogLeNet

from .conv import conv, conv_dw, conv_dw_no_bn
from collections import OrderedDict

import os
import logging

logger = logging.getLogger(__name__)


class CPM(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.pool_center_lower = nn.AvgPool2d(kernel_size=9, stride=8, ceil_mode=True)

#        googlenet = torchvision.models.googlenet(pretrained=False)
        googlenet = GoogLeNet()
        googlenet.maxpool3 = nn.Identity()
        self.inception = nn.Sequential(OrderedDict(
            list(googlenet.named_children())[0: 9]))

        self.conv4_3_CPM = conv_dw(512, 256, 3, 1, padding=1)
        self.conv4_4_CPM = conv_dw(256, 256, 3, 1, padding=1)
        self.conv4_5_CPM = conv_dw(256, 256, 3, 1, padding=1)
        self.conv4_6_CPM = conv_dw(256, 256, 3, 1, padding=1)
        self.conv4_7_CPM = conv(256, 128, 3, 1, padding=1, bn=False, relu=False)

        self.conv5_1_CPM = conv_dw(128, 512, 1, 1)
        self.conv5_2_CPM = conv(512, cfg.MODEL.OUT_CHANNELS, 1, stride=1, bn=False, relu=False)

        self.Mconv1_stage2 = conv_dw(128+1+cfg.MODEL.OUT_CHANNELS, 128, 7, 1, padding=3)
        self.Mconv2_stage2 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv3_stage2 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv4_stage2 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv5_stage2 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv6_stage2 = conv_dw(128, 128, 1, 1)
        self.Mconv7_stage2 = conv(128, cfg.MODEL.OUT_CHANNELS, 1, 1, bn=False, relu=False)

        self.Mconv1_stage3 = conv_dw(128+1+cfg.MODEL.OUT_CHANNELS, 128, 7, 1, padding=3)
        self.Mconv2_stage3 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv3_stage3 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv4_stage3 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv5_stage3 = conv_dw(128, 128, 7, 1, padding=3)
        self.Mconv6_stage3 = conv_dw(128, 128, 1, 1)
        self.Mconv7_stage3 = conv(128, cfg.MODEL.OUT_CHANNELS, 1, 1, bn=False, relu=False)

    def _main(self, image):
        x = self.inception(image)
        x = self.conv4_3_CPM(x)
        x = self.conv4_4_CPM(x)
        x = self.conv4_5_CPM(x)
        x = self.conv4_6_CPM(x)
        x = self.conv4_7_CPM(x)

        return x

    def _stage1(self, conv4_7_CPM):
        x = self.conv5_1_CPM(conv4_7_CPM)
        x = self.conv5_2_CPM(x)

        return x

    def _stage2(self, conv4_7_CPM, prev_stage_map, pool_center_lower_map):
        x = torch.cat([conv4_7_CPM, prev_stage_map, pool_center_lower_map], dim=1).float()
        x = self.Mconv1_stage2(x)  # 128
        x = self.Mconv2_stage2(x)
        x = self.Mconv3_stage2(x)
        x = self.Mconv4_stage2(x)
        x = self.Mconv5_stage2(x)
        x = self.Mconv6_stage2(x)
        x = self.Mconv7_stage2(x)

        return x

    def _stage3(self, conv4_7_CPM, prev_stage_map, pool_center_lower_map):
        x = torch.cat([conv4_7_CPM, prev_stage_map, pool_center_lower_map], dim=1).float()
        x = self.Mconv1_stage3(x)  # 128
        x = self.Mconv2_stage3(x)
        x = self.Mconv3_stage3(x)
        x = self.Mconv4_stage3(x)
        x = self.Mconv5_stage3(x)
        x = self.Mconv6_stage3(x)
        x = self.Mconv7_stage3(x)

        return x

    def forward(self, image, center_map):
        pool_center_lower_map = self.pool_center_lower(center_map)
        pool_center_lower_map = pool_center_lower_map.unsqueeze(1)

        conv4_7_CPM = self._main(image)
        conv5_2_CPM_map = self._stage1(conv4_7_CPM)

        Mconv7_stage2_map = self._stage2(
            conv4_7_CPM, conv5_2_CPM_map, pool_center_lower_map)

        Mconv7_stage3_map = self._stage2(
                conv4_7_CPM, Mconv7_stage2_map, pool_center_lower_map)

        if self.training:
            return [conv5_2_CPM_map, Mconv7_stage2_map, Mconv7_stage3_map]
        else:
            return Mconv7_stage3_map

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            new_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    new_dict[k] = v
                elif 'inception.'+k in model_dict:
                    new_dict['inception.'+k] = v
            

            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = CPM(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
