#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:08:52 2020

@author: aiot
"""
import math
from torch.nn.modules.utils import _quadruple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# binary modules

class BinarizeF(Function):
    @staticmethod
    def forward(cxt, _input):
        output = _input.new(_input.size())
        output[_input >= 0] = 1
        output[_input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, _input):
        # aliases
        binarize = BinarizeF.apply
        output = self.hardtanh(_input)
        output = binarize(output)
        return output


class BinLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinLinear, self).__init__(*kargs, **kwargs)

    def forward(self, x):
        scaling_factor = 1
        binarize = BinarizeF.apply
        binary_weights_no_grad = scaling_factor * binarize(self.weight)
        cliped_weights = torch.clamp(self.weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # foward use binary; backward use cliped weights

        y = F.linear(x, binary_weights, bias=None)
        if self.bias is not None:
            y += self.bias.view(1, -1).expand_as(y)

        return y

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


class BinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, x):
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weights), dim=3, keepdim=True),
        #                                        dim=2, keepdim=True),
        #                             dim=1, keepdim=True)
        #scaling_factor = scaling_factor.detach()
        scaling_factor = 1
        binarize = BinarizeF.apply
        binary_weights_no_grad = scaling_factor * binarize(self.weight)
        cliped_weights = torch.clamp(self.weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # foward use binary; backward use cliped weights
        x=F.pad(x, pad=_quadruple(1), mode='replicate')
        x = F.pad(x, (1, 1), "constant", 0)
        y = F.conv2d(x, binary_weights, bias=None, stride=self.stride, padding=0, groups=self.groups)

        return y


class BinConv1d(nn.Conv1d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv1d, self).__init__(*kargs, **kwargs)

    def forward(self, x):

        scaling_factor = 1
        binarize = BinarizeF.apply
        binary_weights_no_grad = scaling_factor * binarize(self.weight)
        cliped_weights = torch.clamp(self.weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # foward use binary; backward use cliped weights

        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups, bias=None)

        return y
class Conv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                 padding=(1, 1, 1, 1), padding_mode='replicate', bias=False,
                 stride=1, groups=1, **kwargs):
        #scaling_factor=0.1,
        super(Conv_Block, self).__init__()

        self.groups = groups
        #self.scaling_factor = scaling_factor
        self.conv = BinConv2d(in_planes, out_planes, kernel_size=kernel_size,
                              padding=padding, padding_mode=padding_mode,
                              stride=stride, bias=bias, groups=self.groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.hardtanh = BinaryTanh()

    def forward(self, x):

        y = self.conv(x)
        y = self.bn(y)
        y = self.hardtanh(y)
        return y


class Conv_Block_multibit(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                 padding=1, padding_mode='replicate', bias=False,
                 stride=1, groups=1, **kwargs):
        #scaling_factor=0.1,
        super(Conv_Block_multibit, self).__init__()

        self.groups = groups
        self.Conv2d=conv2d_Q_fn(w_bit=2)
        self.conv=self.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=bias, groups=self.groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = activation_quantize_fn(a_bit=2)

    def forward(self, x):

        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return y

def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** (k-1))
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, -0.5, 0.5))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

# class Linear_Block(nn.Module):

class End_FC_Block(nn.Module):
    def __init__(self, in_planes, num_classes, bias=True,
                 scaling_factor=1, **kwargs):
        super(End_FC_Block, self).__init__()
        self.linear = BinLinear(in_planes, num_classes, bias=True)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        y = self.linear(x)
        y = self.bn(y)
        return y
