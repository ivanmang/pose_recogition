# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from models.binary_module import BinConv2d, BinLinear, BinaryTanh


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.conv = BinConv2d(in_planes, growth_rate, kernel_size=3, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.hardtanh = BinaryTanh()
        # nn.init.constant(self.conv.weight,0)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.hardtanh(out)
        out = torch.cat([out, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, kernel_size=None, padding=None):
        super(Transition, self).__init__()
        if kernel_size is None:
            kernel_size = 3
            padding = 1
        self.conv = BinConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.hardtanh = BinaryTanh()
        # nn.init.constant(self.conv.weight,0)

    def forward(self, x):
        #        out = F.max_pool2d(x, 2)
        out = self.conv(x)
        out = self.bn(out)
        out = self.hardtanh(out)

        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, height=32, width=32, growth_rate=16,
                 reduction=1, input_channel=1, num_classes=3):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 4 * growth_rate
        self.conv1 = nn.Sequential(
            BinConv2d(input_channel, num_planes, kernel_size=(3,3), padding=0, dilation=1, stride=1, bias=False),
            nn.BatchNorm2d(num_planes),
            BinaryTanh()
        )

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes


        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        if len(nblocks) > 2:
            self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
            num_planes += nblocks[2] * growth_rate
            out_planes = int(math.floor(num_planes * reduction))
            self.trans3 = Transition(num_planes, out_planes)
            num_planes = out_planes
        #
        # if len(nblocks) > 3:
        #     self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        #     num_planes += nblocks[3] * growth_rate
            # out_planes = int(math.floor(num_planes * reduction))
            # self.trans4 = Transition(num_planes, out_planes)
            # num_planes = out_planes

        # self.dense5 = self._make_dense_layers(block, num_planes, nblocks[4])
        # num_planes += nblocks[4] * growth_rate

        # calculate size
        h, w = height, width
        h, w = np.ceil(h/2/2/2), np.ceil(w/2/2/2)
        cnn_final_fm = int(h*w)

        self.classifier = nn.Sequential(
            BinLinear(num_planes*cnn_final_fm, num_classes),
            nn.BatchNorm1d(num_classes),
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, 1, :, :]
        x=torch.unsqueeze(x, 1)
        out=self.conv1(x)
        out=self.dense1(out)
        out=self.trans1(out)
        out=self.dense2(out)
        out=self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        # out = self.dense4(out)


        # out = self.trans01(self.trans0(self.conv1(x)))

        out = out.view(out.size(0), -1)

        out = self.classifier(out)

        return out


