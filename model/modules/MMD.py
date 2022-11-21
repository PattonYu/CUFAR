import torch

class MMD:
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        # exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                    bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n] 
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1) 
        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1)

        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1)
        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1) 
            
        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss