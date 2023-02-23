import torch
import torch.nn as nn
from einops import rearrange

class mini_model(nn.Module):
    def __init__(self, n_channel, scale_factor, in_channel, kernel_size, padding, groups):
        super(mini_model, self).__init__()
        self.n_channels = n_channel
        self.scale_factor = scale_factor
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv1 = nn.Conv2d(self.in_channel, self.n_channels, self.kernel_size, 1, padding, 
                                    groups= self.groups)
        self.relu = nn.ReLU(inplace= True)
        self.conv2 = nn.Conv2d(self.n_channels, self.scale_factor ** 2 * self.n_channels, 
                                    3, 1, 1, groups= self.groups)

        self.pixelshuffle = nn.PixelShuffle(upscale_factor= self.scale_factor)

        self.ic_layer = IC_layer(self.n_channels, 0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x))
        x = self.relu(self.pixelshuffle(x))
        x = self.ic_layer(x)
        return x


class IC_layer(nn.Module):
    def __init__(self, n_channel, drop_rate):
        super(IC_layer, self).__init__()
        self.batch_norm = nn.BatchNorm2d(n_channel)
        self.drop_rate = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.drop_rate(x)
        return x


class CUFAR(nn.Module):
    def __init__(self, height=32, width=32, use_exf=True, scale_factor=4, 
                    channels=128, sub_region = 4, scaler_X=1, scaler_Y=1, args= None):
        super(CUFAR, self).__init__()
        self.height = height
        self.width = width
        self.use_exf = use_exf
        self.n_channels = channels
        self.scale_factor = scale_factor
        self.out_channel = 1
        self.sub_region = sub_region
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.args = args
        time_span = 24 if args.dataset == 'TaxiNYC' else 15
        if use_exf:
            self.time_emb_region = nn.Embedding(time_span, int((self.width * self.height)/(self.sub_region ** 2)))
            self.time_emb_global = nn.Embedding(time_span, (self.width * self.height))

            # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_day = nn.Embedding(8, 2)
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(12, 64),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, int((self.width * self.height)/(self.sub_region ** 2))),
                nn.ReLU(inplace=True)
            )

            self.ext2lr_global = nn.Sequential(
                nn.Linear(12, 64),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, int(self.width * self.height)),
                nn.ReLU(inplace=True)
            )
            self.global_model = mini_model(self.n_channels, self.scale_factor, 3, 9, 4, 1)
            self.local_sub_model = mini_model(self.n_channels * (sub_region ** 2), 
                                self.scale_factor, 3 * (sub_region **2), 3, 1, sub_region **2)
        else:
            self.global_model = mini_model(self.n_channels, self.scale_factor, 1, 9, 4, 1)
            self.local_sub_model = mini_model(self.n_channels * (sub_region ** 2), 
                                self.scale_factor, 1 * (sub_region **2), 3, 1, sub_region **2)
        self.relu = nn.ReLU()
        time_conv = []
        for i in range(time_span):
            time_conv.append(nn.Conv2d(128*2, self.out_channel, 3, 1, 1))
        self.time_conv = nn.Sequential(*time_conv)
    

    def embed_ext(self, ext):
        ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 5].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(
            ext[:, 6].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, :4]

        return torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)


    def normalization(self, x, save_x):
        w = (nn.AvgPool2d(self.scale_factor)(x)) * self.scale_factor ** 2
        w = nn.Upsample(scale_factor= self.scale_factor, mode='nearest')(w)
        w = torch.divide(x, w + 1e-7)
        up_c = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(save_x)
        x = torch.multiply(w, up_c)
        return x


    def forward(self, x, eif):
        save_x = x
        if self.use_exf:
            x = rearrange(x, 'b c (ph h) (pw w) -> (ph pw) b c h w', ph= self.sub_region, pw= self.sub_region)
            ext_emb = self.embed_ext(eif)
            t = eif[:, 5].long().view(-1, 1)
            if self.args.dataset == 'TaxiBJ':
                t -= 7
            time_emb_region = self.time_emb_region(t).view(-1, 1, 
                                                        int(self.height/self.sub_region), 
                                                        int(self.width/self.sub_region))
            time_emb_global = self.time_emb_global(t).view(-1, 1, 
                                                        self.height, self.width)
            ext_out = self.ext2lr(ext_emb).view(-1, 1, int(self.width/self.sub_region), 
                                                    int(self.height/self.sub_region))
            ext_out_global = self.ext2lr_global(ext_emb).view(-1, 1, self.width, self.height)
            output_x = list(map(lambda x: torch.cat([x, ext_out, time_emb_region], dim=1).unsqueeze(0) ,x))
            output_x = torch.cat(output_x, dim=0)
            local_c = rearrange(output_x, '(ph pw) b c h  w -> b (ph pw c) h w', 
                                        ph= self.sub_region, pw= self.sub_region)
            output = self.local_sub_model(local_c)
            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)', 
                                        ph= self.sub_region, pw= self.sub_region)
            global_f = self.global_model(torch.cat([save_x, ext_out_global, time_emb_global], dim= 1))
        else:
            local_c = rearrange(x, 'b c (ph h) (pw w) -> b (ph pw c) h w', 
                                        ph= self.sub_region, pw= self.sub_region)
            output = self.local_sub_model(local_c)
            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)', 
                                        ph= self.sub_region, pw= self.sub_region)
            global_f = self.global_model(save_x)

        x = torch.cat([local_f, global_f], dim= 1)

        output= []
        for i in range(x.size(0)):
            t = int(eif[i, 5].cpu().detach().numpy())
            if self.args.dataset == 'TaxiBJ':
                t -= 7
            output.append(self.relu(self.time_conv[t](x[i].unsqueeze(0))))
        x = torch.cat(output, dim= 0)
        x = self.normalization(x, save_x * self.scaler_X / self.scaler_Y)
        
        return x