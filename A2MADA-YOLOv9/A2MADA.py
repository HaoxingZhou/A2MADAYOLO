from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

#梯度反转层
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

#抓取字典中的6, 8, 10, 21, 23, 25
def extract_layer_outputs(captured_outputs):
    # 提取特定层的输出
    layer_4_output = captured_outputs.get(4, None)
    layer_6_output = captured_outputs.get(6, None)
    layer_8_output = captured_outputs.get(8, None)
    layer_15_output = captured_outputs.get(15, None)
    layer_18_output = captured_outputs.get(18, None)
    layer_21_output = captured_outputs.get(21, None)

    return layer_4_output, layer_6_output, layer_8_output, layer_15_output, layer_18_output, layer_21_output

#基于对抗的域自适应方法
class A2MADA(nn.Module):
    def __init__(self, device):
        super(A2MADA, self).__init__()
        self.da_img_model_6 = DA_Img_WithAtt(512).to(device)
        self.da_img_model_8 = DA_Img_WithAtt(512).to(device)
        self.da_img_model_10 = DA_Img_WithAtt(512).to(device)
        self.da_ins_model_21 = DAInsHead(256).to(device)
        self.da_ins_model_23 = DAInsHead(512).to(device)
        self.da_ins_model_25 = DAInsHead(512).to(device)

    def forward(self, captured_src, captured_tgt):
        layers_src = extract_layer_outputs(captured_src)
        layers_tgt = extract_layer_outputs(captured_tgt)

        da_img_loss = 0
        da_ins_loss = 0
        da_consist_loss = 0
        attention_loss = 0

        da_img_models = [self.da_img_model_6, self.da_img_model_8, self.da_img_model_10]
        da_ins_models = [self.da_ins_model_21, self.da_ins_model_23, self.da_ins_model_25]

        src_img_consist_outputs = []
        tgt_img_consist_outputs = []
        src_ins_consist_outputs = []
        tgt_ins_consist_outputs = []

        for i, (src_layer, tgt_layer) in enumerate(zip(layers_src[:3], layers_tgt[:3])):
            src_img_rev = ReverseLayerF.apply(src_layer, 0.001)
            tgt_img_rev = ReverseLayerF.apply(tgt_layer, 0.001)
            src_img_consist = ReverseLayerF.apply(src_layer, -0.001)
            tgt_img_consist = ReverseLayerF.apply(tgt_layer, -0.001)

            da_img_model = da_img_models[i]
            src_grl_output, src_att_maps = da_img_model(src_img_rev)
            tgt_grl_output, tgt_att_maps = da_img_model(tgt_img_rev)
            src_img_consist_output, _ = da_img_model(src_img_consist)
            tgt_img_consist_output, _ = da_img_model(tgt_img_consist)
            src_img_consist_output = src_img_consist_output.sigmoid()
            tgt_img_consist_output = tgt_img_consist_output.sigmoid()

            src_img_consist_outputs.append(src_img_consist_output)
            tgt_img_consist_outputs.append(tgt_img_consist_output)

            for src_att_map, tgt_att_map in zip(src_att_maps, tgt_att_maps):
                attention_loss += F.mse_loss(src_att_map, tgt_att_map)
        src_img_consist_mean = process_outputs(src_img_consist_outputs)
        tgt_img_consist_mean = process_outputs(tgt_img_consist_outputs)

        for i, (src_layer, tgt_layer) in enumerate(zip(layers_src[3:], layers_tgt[3:])):
            src_ins_rev = ReverseLayerF.apply(src_layer, 0.001)
            tgt_ins_rev = ReverseLayerF.apply(tgt_layer, 0.001)
            src_ins_consist = ReverseLayerF.apply(src_layer, -0.001)
            tgt_ins_consist = ReverseLayerF.apply(tgt_layer, -0.001)

            da_ins_model = da_ins_models[i]
            src_ins_rev = F.adaptive_avg_pool2d(src_ins_rev, (1, 1)).squeeze()
            tgt_ins_rev = F.adaptive_avg_pool2d(tgt_ins_rev, (1, 1)).squeeze()
            src_ins_output = da_ins_model(src_ins_rev)
            tgt_ins_output = da_ins_model(tgt_ins_rev)
            src_ins_consist_output = da_ins_model(F.adaptive_avg_pool2d(src_ins_consist, (1, 1)).squeeze()).sigmoid()
            tgt_ins_consist_output = da_ins_model(F.adaptive_avg_pool2d(tgt_ins_consist, (1, 1)).squeeze()).sigmoid()

            src_ins_consist_outputs.append(src_ins_consist_output)
            tgt_ins_consist_outputs.append(tgt_ins_consist_output)

            da_source_label = torch.zeros_like(src_ins_output).type(torch.cuda.FloatTensor)
            da_target_label = torch.ones_like(tgt_ins_output).type(torch.cuda.FloatTensor)
            grl_s_loss = F.binary_cross_entropy_with_logits(src_ins_output, da_source_label)
            grl_t_loss = F.binary_cross_entropy_with_logits(tgt_ins_output, da_target_label)
            da_ins_loss += grl_s_loss + grl_t_loss


        src_ins_consist_mean = torch.mean(torch.cat(src_ins_consist_outputs, dim=0), dim=0, keepdim=True)
        tgt_ins_consist_mean = torch.mean(torch.cat(tgt_ins_consist_outputs, dim=0), dim=0, keepdim=True)


        da_consist_loss = consistency_loss(src_img_consist_mean, src_ins_consist_mean, tgt_img_consist_mean,
                                           tgt_ins_consist_mean, size_average=True)

        # 最终平均化da_img_loss和da_ins_loss
        da_img_loss /= 6
        da_ins_loss /= 6
        attention_loss = attention_loss / 6

        da_loss = (da_img_loss + da_ins_loss) / 2 + 0.2 * da_consist_loss + 0.05 * attention_loss
        return da_loss


#DA_Img 图像级域自适应
class DA_Img_WithAtt(nn.Module):
    def __init__(self, in_channels):
        super(DA_Img_WithAtt, self).__init__()
        # 初始化层时，将卷积、BN和注意力层分别存储
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.cas = nn.ModuleList()  # 注意力层

        # 根据输入通道数选择不同的过渡逻辑
        if in_channels == 512:
            channels = [512, 256, 128, 1]
        elif in_channels == 256:
            channels = [256, 128, 64, 1]
        else:
            raise ValueError("Unsupported in_channels")

        for i in range(len(channels) - 1):
            self.convs.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, stride=1))
            self.bns.append(nn.BatchNorm2d(channels[i + 1]))
            if i < len(channels) - 2:  # 最后一个卷积后不加注意力层
                self.cas.append(ChannelAttention(channels[i + 1]))

            # 初始化权重和偏置
            nn.init.normal_(self.convs[-1].weight, std=0.1)
            nn.init.constant_(self.convs[-1].bias, 0)

    def forward(self, x):
        x = x.to(torch.float32)
        attention_maps = []

        for conv, bn, ca in zip(self.convs, self.bns, self.cas):
            x = conv(x)
            x = bn(x)
            x, attention_map = ca(x)  # 注意力层返回特征图和注意力图
            attention_maps.append(attention_map)
            x = F.leaky_relu(x)

        # 处理最后一个卷积层，它后面没有注意力层
        x = self.convs[-1](x)
        x = self.bns[-1](x)
        x = F.leaky_relu(x)

        img_features = x.view(-1, 1)
        return img_features, attention_maps

#DAInsHead
class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


def consistency_loss(src_img, src_ins, tgt_img, tgt_ins, size_average=True):
    """
    Calculate consistency loss between image-level and instance-level classifier outputs
    for both source and target domains.
    """
    # 计算源域的一致性损失
    src_loss = torch.abs(src_img - src_ins)
    # 计算目标域的一致性损失
    tgt_loss = torch.abs(tgt_img - tgt_ins)
    # 合并两个域的损失
    loss = (src_loss + tgt_loss) / 2
    if size_average:
        return loss.mean()
    return loss.sum()

# # 定义一个函数来处理输出张量
# def process_outputs(outputs):
#     processed_outputs = []
#     for output in outputs:
#         # 计算每个输出的平均值
#         output_mean = torch.mean(output, dim=0, keepdim=True)  # 沿着第一个维度计算平均值
#         processed_outputs.append(output_mean)
#     return processed_outputs

def process_outputs(outputs):
    # 计算每个输出的平均值，并收集这些平均值
    processed_outputs = [torch.mean(output, dim=0, keepdim=True) for output in outputs]

    # 计算所有平均值的整体平均值
    overall_mean = torch.mean(torch.cat(processed_outputs, dim=0), dim=0)

    return overall_mean

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 确保 reduction_ratio 不会导致除法结果为0
        reduced_channels = max(in_channels // reduction_ratio, 1)  # 至少为1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention, attention
