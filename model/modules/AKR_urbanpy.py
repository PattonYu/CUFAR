import torch
import torch.nn.functional as F
import random
from model.modules.augments import apply_augment
from copy import deepcopy
from model.modules.MMD import MMD
from src.utils import get_gt_densities
from model.modules.urbanpy_layers import batch_kl
import numpy as np
import math

class continual:
    def __init__(self, model, buffer, n_tasks, args):
        self.model = model
        self.n_task = n_tasks
        self.buffer = buffer
        self.args = args
        self.optimizer = torch.optim.Adam(
                    model.parameters(), lr= args.lr, betas=(args.b1, args.b1))
        self.criterion = F.mse_loss
        self.flow_criterion = torch.nn.L1Loss()
        self.train_data_len = 0
        self.MMD = MMD()
        self.scales = [2**(i+1) for i in range(args.N)]

    def draw_batches(self, flows, t):
        if t > 1:
            buffer_flows = self.buffer.get_data(self.args.minibatch_size, t)
            expand_flows = list()
            for i in range(len(flows)):
                expand_flows.append(torch.cat([flows[i], buffer_flows[i]], dim= 0))
            
            expand_flows[-2], expand_flows[0], mask, aug = apply_augment(fine= expand_flows[-2],
                                                                        coarse= expand_flows[0])

            alpha = self.MMD.forward(expand_flows[0].view(expand_flows[0].size(0), -1).cpu(), 
                                        flows[0].view(flows[0].size(0), -1).cpu())
            alpha = 2 - (2/(1+math.exp(-alpha*3)))
            return expand_flows, alpha, mask, aug
        else:
            return flows, 1, 0, 0

    def observe(self, flows, t):
        # Reservoir sampling memory update:
        self.buffer.add_data(flows, t)
        buffer_flows, alpha, mask, aug = self.draw_batches(flows, t)
        
        flows = buffer_flows[:-1]; ext = buffer_flows[-1]
        gt_dens, gt_masks = get_gt_densities(flows, self.args)
        self.model.train()
        self.optimizer.zero_grad()
        self.train_data_len += flows[0].size(0)
        if t > 1:
            weights_before = deepcopy(self.model.state_dict())
        densities, outs = self.model(flows[0], ext)
        if aug == "cutout":
            outs[-1], flows[-1] = outs[-1]*mask, flows[-1]*mask
        loss_mse, losses = self.compute_loss(predicts=outs, ys=flows[1:], weights=self.args.loss_weights)
        loss_kl, losses_kl = self.compute_kl_loss(predicts=densities[1:], ys=gt_dens, 
                                            scales=self.scales, masks=gt_masks)
        loss = (1-self.args.alpha)*loss_mse + self.args.alpha*loss_kl
        loss.backward()
        self.optimizer.step()
        if t > 1:
            weights_after = self.model.state_dict()
            model_dict = {}
            for name in weights_after: 
                model_dict[name] = weights_before[name] + ((weights_after[name] - weights_before[name]) * alpha)
            self.model.load_state_dict(model_dict)

        return loss.item() * flows[2].size(0)

    # Loss functions
    def compute_loss(self, predicts, ys, weights=[1,1,1]):
        batch_size = len(predicts[0])
        assert len(predicts) == len(ys),\
                'out len: {}, flow len: {}'.format(len(predicts), len(ys))
        losses = [self.criterion(yhat, y)*weights[i] 
                for i, (yhat, y) in enumerate(zip(predicts, ys))]
        return sum(losses), torch.sqrt(torch.stack(losses)).data.cpu().numpy()

    def compute_kl_loss(self, predicts, ys, masks, scales, weights=[1,1,1,1]):
        losses = [batch_kl(yhat, y, scale, mask)*weights[i]
                for i, (yhat, y, scale, mask) in enumerate(zip(predicts, ys, scales, masks))]
        return sum(losses), torch.stack(losses).detach().cpu().numpy()