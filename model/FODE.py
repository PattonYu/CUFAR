import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules.ODE import *

class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):  # 4
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)  # k_size = 4  #(16,1,32,32)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')  # (16,1,128,128)
        self.epsilon = 1e-6


    def forward(self, x):
        out = self.avgpool   (x) * self.upscale_factor ** 2  # sum pooling
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


class FODE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16, use_ode=True,
                 base_channels=128, img_width=32, img_height=32, ext_flag=True, scaler_X=1, scaler_Y=1):
        super(FODE, self).__init__()
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
                nn.Linear(12, 64), 
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

        # ODE
        ode_blocks = []
        for _ in range(1):
            ode_blocks.append(ODEBlock(ODEfunc(base_channels)))
        self.ode_blocks = nn.Sequential(*ode_blocks)

        ode_blocks1 = []
        for _ in range(1):
            ode_blocks1.append(ODEBlock(ODEfunc_ext(32)))
        self.ode_blocks1 = nn.Sequential(*ode_blocks1)


        ode_blocks2 = []
        for _ in range(1):
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
        # self.se1 = SE(base_channels)

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

            #ext_out = self.ext2lr(torch.cat([day,hour,weekend,ext[:,:3],weather], dim=1)).view(-1, 1, self.img_width, self.img_height)
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
        inp4 = out3
        #inp4 = torch.add(out_f, out3)

        out4 = self.upsampling(inp4)  # (16,128,32,32) ====> (16,128,128,128)
        #out4 = self.se1(out4)

        #concatenation backward
        if self.ext_flag:
            ext_out = self.ext2hr(ext_out)                          #(16,1,32,32)===>(16,1,128,128)
            ext_out1 = self.ode_blocks2(ext_out)
            ext_out = torch.add(ext_out, ext_out1)
            #out = self.conv3(torch.cat([out4, ext_out1], dim=1))
            out = self.conv3(out4)
            out = torch.add(out, ext_out)
            ext_out = self.den_softmax(ext_out1)
        else:
            out = self.conv3(out4)  # (16,129,128,128)===>(16,1,128,128)

        #get the distribution matrix
        out = self.den_softmax(out)                                  #(16,1,128,128)
        # n2 = out
        #out = torch.add(0.99*out,0.01*ext_out)  # a =0.99  b = 0.01
        if self.ext_flag:
            out = out.mul(torch.sigmoid(ext_out)) + out
            out = self.den_softmax(out)
        # fine = out
        #recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)          #x*150
        return out