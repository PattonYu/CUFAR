import torch
import torch.nn.functional as F
from model.modules.augments import apply_augment
from copy import deepcopy
from model.modules.MMD import MMD
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
        self.train_data_len = 0
        self.MMD = MMD()

    def draw_batches(self, c_map, f_map, exf, t):
        if t > 1:
            buffer_c_map, buffer_f_map, buffer_exf = self.buffer.get_data(self.args.minibatch_size, t)

            exp_c_map = torch.cat([c_map, buffer_c_map], dim= 0)
            exp_f_map = torch.cat([f_map, buffer_f_map], dim= 0)
            replay_exf = torch.cat([exf, buffer_exf], dim= 0)

            buffer_f_map, buffer_c_map, mask, aug = apply_augment(exp_f_map, exp_c_map)

            alpha = self.MMD.forward(buffer_c_map.view(buffer_c_map.size(0), -1).cpu(),
                                        c_map.view(c_map.size(0), -1).cpu())
            alpha = 2 - (2/(1+math.exp(-alpha*3)))

            return buffer_c_map, buffer_f_map, replay_exf, alpha, mask, aug
        else:
            return c_map, f_map, exf, 1, 0, 0

    def observe(self, c_maps, f_maps, exf, t):
        # Reservoir sampling memory update:
        self.buffer.add_data(c_maps, f_maps, exf, t)
        replay_c_maps, replay_f_maps, replay_exf, alpha, mask, aug = self.draw_batches(c_maps, f_maps, exf, t)

        self.model.train()
        self.optimizer.zero_grad()
        self.train_data_len += c_maps.size(0)

        # save theta_0
        weights_before = deepcopy(self.model.state_dict())
        
        pred_f_map = self.model(replay_c_maps, replay_exf)
        if aug == "cutout":
            pred_f_map, replay_f_maps = pred_f_map*mask, replay_f_maps*mask       
        loss = self.criterion(pred_f_map, replay_f_maps)
        loss.backward()
        self.optimizer.step()

        # get theta_1
        weights_after = self.model.state_dict()
        model_dict = {}
        for name in weights_after: 
            model_dict[name] = weights_before[name] + ((weights_after[name] - weights_before[name]) * alpha)

        # update
        self.model.load_state_dict(model_dict)

        return loss.item() * pred_f_map.size(0)