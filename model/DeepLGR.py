import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2 # sum pooling
        out = self.upsample(out)
        return torch.div(x, out + self.epsilon)


class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DeepLGR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=12,
                 base_channels=128, img_width=32, img_height=32, ext_flag=True, scaler_X=1500, scaler_Y=100,predictor='td'):
        super(DeepLGR, self).__init__()
        self.ext_flag = ext_flag
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height
        self.n_channels = base_channels
        self.predictor= predictor

        if ext_flag:
            self.embed_day = nn.Embedding(8, 2) # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3) # hour range [0, 23]
            self.embed_weather = nn.Embedding(18, 3) # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, img_width * img_height),
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
            conv3_in = in_channels + 1
        else:
            conv1_in = in_channels
            conv3_in = base_channels

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, base_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )
        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_in, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks n_residual_blocks
        se_blocks = []
        for _ in range(n_residual_blocks):
            se_blocks.append(SEBlock(base_channels))
        self.senet = nn.Sequential(*se_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.BatchNorm2d(base_channels))

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

        # if flag_global:
        self.glonet = GlobalNet(base_channels, base_channels, (1, 2, 4, 8), img_height*4, img_width*4)
        
        # specify predictor

        if predictor == 'td': # tensor decomposition
            d1 = 16
            d2 = 16
            d3 = 32
            self.core = nn.Parameter(torch.FloatTensor(d1, d2, d3)) 
            self.F = nn.Parameter(torch.FloatTensor(d3, base_channels)) 
            self.H = nn.Parameter(torch.FloatTensor(d1, img_height*4))
            self.W = nn.Parameter(torch.FloatTensor(d2, img_width*4))
            nn.init.normal_(self.core, 0, 0.02)
            nn.init.normal_(self.F, 0, 0.02)
            nn.init.normal_(self.H, 0, 0.02)
            nn.init.normal_(self.W, 0, 0.02)
        elif predictor == 'md': # matrix factorization
            self.L = nn.Parameter(torch.FloatTensor(img_width*4 * img_height*4, 10))
            self.R = nn.Parameter(torch.FloatTensor(10, base_channels))
            nn.init.normal_(self.L, 0, 0.02)
            nn.init.normal_(self.R, 0, 0.02)
        else:
            self.output_conv = nn.Sequential(nn.Conv2d(base_channels, 1, 1, 1, 0))

    def forward(self, x, ext):
        inp = x
        b = x.shape[0]
        if self.ext_flag:
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)
            inp = torch.cat([x, ext_out], dim=1)

        out1 = self.conv1(inp)
        out = self.senet(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)

        # glonet
        out = self.glonet(out) # [b, n_filters, H, W]

        # predictor
        if self.predictor == 'td': # tensor decomposition
            out = out.reshape(b, self.n_channels, -1).permute(0, 2, 1) # [b, H*W, n_filters]
            region_param = torch.matmul(self.core, self.F) # [16, 16, n_filters*out_channels]
            region_param = region_param.permute(1, 2, 0) # [16, n_f*out_c, 16]
            region_param = torch.matmul(region_param, self.H) # [16, n_f*out_c, H]
            region_param = region_param.permute(1, 2, 0) # [n_f*out_c, H, 16]
            region_param = torch.matmul(region_param, self.W) # [n_f*out_c, H, W]
            region_param = region_param.unsqueeze(0).repeat(b, 1, 1, 1) # [b, n_f*out_c, H, W]
            region_param = region_param.reshape(b, -1, self.n_channels, 128**2).permute(0, 3, 2, 1) # [b, H*W, n_f, 2]
            region_features = out.unsqueeze(3).repeat(1, 1, 1, 1) # [b, H*W, n_filters, 2]
            out = torch.sum(region_features * region_param, 2).reshape(b, 128, 128, -1)  # [b, H, W, 2]
            out = out.permute(0, 3, 1, 2) 
        elif self.predictor == 'md': # matrix decomposition
            out = out.reshape(b, self.n_channels, -1).permute(0, 2, 1) # [b, H*W, n_filters]
            region_param = torch.matmul(self.L, self.R).unsqueeze(0) # [1, H*W, n_filter*2]
            region_param = region_param.repeat(b, 1, 1).reshape(b, -1, self.n_channels, 1) # [b, H*W, n_filter, 2]
            region_features = out.unsqueeze(3).repeat(1, 1, 1, 1) # [b, H*W, n_filters, 2]
            out = torch.sum(region_features * region_param, 2).reshape(b, 128, 128, -1)  # [b, H, W, 2]
            out = out.permute(0, 3, 1, 2)
        

        # concatenation backward
        if self.ext_flag:
            ext_out = self.ext2hr(ext_out)
            # try:
            out = self.conv3(torch.cat([out, ext_out], dim=1))
            # except RuntimeError as e:
            #     # print('OUt shape', out.shape, 'ext shape', ext_out.shape, e)
            #     pass
        else:
            out = self.conv3(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)
        return out

class GlobalNet(nn.Module):
    def __init__(self, features=64, out_features=64, sizes=(1, 2, 4, 8), height=128, width=128):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features + features // 8 * 4, out_features, kernel_size=1)
        self.relu = nn.ReLU()

        self.deconvs = nn.ModuleList()
        for size in sizes:
            self.deconvs.append(SubPixelBlock(features // 8, upscale_factor=height // size))
        
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features // 8, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(features // 8)
        return nn.Sequential(prior, conv, bn)

    def forward(self, x):
        priors = [upsample(stage(x)) for stage, upsample in zip(self.stages, self.deconvs)]
        out = priors + [x]
        bottle = self.bottleneck(torch.cat(out, 1))
        return self.relu(bottle)

class SubPixelBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(SubPixelBlock, self).__init__()
        self.r = upscale_factor
        out_channels = in_channels * upscale_factor * upscale_factor
        self.conv = nn.Conv2d(in_channels,  out_channels, 1, 1, 0)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # x: [b, c, h, w]
        x = self.conv(x) # [b, c*r^2, h, w]
        out = self.ps(x) # [b, c, h*r, w*r]
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, in_features):
        super(SEBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.se = SELayer(in_features)
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        return x + out