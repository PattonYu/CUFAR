import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules.ODE import *
from torchvision.models import vgg19
import math

class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):  # 4
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)  # k_size = 4  #(16,1,32,32)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')  # (16,1,128,128)
        self.epsilon = 1e-7


    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2  # sum pooling
        out = self.upsample(out)
        out = torch.div(x, out + self.epsilon)
        return out

class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)   #x:density  out: lr_img


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp0 = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.mlp1 = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp0(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp1(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SE, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
                 conv=default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1 - i / 10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        # theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base, 1, dim=0)
        # patch size for matching
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        # build feature pyramid
        for i in range(len(self.scale)):
            ref = input
            if self.scale[i] != 1:
                ref = F.interpolate(input, scale_factor=self.scale[i], recompute_scale_factor=True)
            # feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            # sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                            strides=[self.stride, self.stride],
                                            rates=[1, 1],
                                            padding='same')  # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            # feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            # sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            # group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
            # normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi
            # matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi * self.softmax_scale, dim=1)

            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
            y.append(yi)

        y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
        return y


def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class UrbanODE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16, 
                 base_channels=128, ext_dim=7, img_width=32, img_height=32, 
                 ext_flag=True, scaler_X=1500, scaler_Y=100):
        super(UrbanODE, self).__init__()
        self.ext_flag = ext_flag
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        if ext_flag:
            self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(12, 64),  # TODO
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(64, img_width * img_height),
                nn.ReLU(inplace=True)
            )

            self.ext2hr = nn.Sequential(
                nn.Conv2d(1, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
            )

        if ext_flag:
            conv1_in = in_channels + 1
            conv3_in = base_channels
        else:
            conv1_in = in_channels
            conv3_in = base_channels

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, base_channels, 9, 1, 4),
            #nn.ReLU(inplace=True)
        )
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.LayerNorm([base_channels, 32, 32]),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_in, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # input conv
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, base_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )
        # Second conv layer post residual blocks
        self.conv5 = nn.Sequential(
            #nn.LayerNorm([base_channels, 32, 32]),
            #nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # output conv
        self.conv6 = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        self.msa = PyramidAttention(channel=base_channels)

        # ODE
        ode_blocks = []
        for _ in range(1):
            ode_blocks.append(ODEBlock(ODEfunc(base_channels)))
            ode_blocks.append(self.msa)
            ode_blocks.append(ODEBlock(ODEfunc(base_channels)))
        self.ode_blocks = nn.Sequential(*ode_blocks)

        ode_blocks1 = []
        for _ in range(4):
            ode_blocks1.append(ODEBlock(ODEfunc_ext(32)))
        self.ode_blocks1 = nn.Sequential(*ode_blocks1)


        ode_blocks2 = []
        for _ in range(4):
            ode_blocks2.append(ODEBlock(ODEfunc_ext(128)))
        self.ode_blocks2 = nn.Sequential(*ode_blocks2)

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)
        self.se = SE(base_channels)
        self.se1 = SE(base_channels)

    def forward(self, x, ext):
        inp = x

        if self.ext_flag:  #[temp,ws,hol,wen,day,hour,weather]
            # TaxiBJ
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], 
                            dim=1)).view(-1, 1,   self.img_width, self.img_height)

            ext_out1 = self.ode_blocks1(ext_out)
            ext_out = torch.add(ext_out, ext_out1)
            inp = torch.cat([x, ext_out], dim=1)  # inp shape(16,2,32,32)

        out_f = self.conv1(inp)  # (16,2,32,32) ====> (16,128,32,32)

        #ODEDB
        out1 = self.ode_blocks(out_f)
        inp2 = torch.add(out_f, out1)
        out2 = self.conv2(inp2)
        inp3 = torch.add(torch.add(out_f, out1), out2)
        out3 = self.se(inp3)
        inp4 = torch.add(out_f, out3)

        out4 = self.upsampling(inp4)  # (16,128,32,32) ====> (16,128,128,128)
        out4 = self.se1(out4)

        #concatenation backward
        if self.ext_flag:
            ext_out = self.ext2hr(ext_out)                          #(16,1,32,32)===>(16,1,128,128)
            # before = ext_out
            ext_out1 = self.ode_blocks2(ext_out)
            # after = ext_out1
            ext_out = torch.add(ext_out, ext_out1)
            #out = self.conv3(torch.cat([out4, ext_out], dim=1))
            out = self.conv3(out4)
            out = torch.add(out, ext_out)
        else:
            out = self.conv3(out4)  # (16,129,128,128)===>(16,1,128,128)

        #get the distribution matrix
        out = self.den_softmax(out)                                  #(16,1,128,128)
        #recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)          #x*150
        return out

