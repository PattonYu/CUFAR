import os
import sys
import warnings
import numpy as np
import random
import warnings
import time
import torch
import torch.nn as nn
from src.metrics import get_MAE, get_MSE, get_MAPE
from src.utils import print_model_parm_nums, get_lapprob_dataloader
from model.UrbanPy import UrbanPy, weights_init_normal
from model.modules.urbanpy_layers import batch_kl
from model.modules.AKR_urbanpy import continual
from src.args import get_args
from model.modules.memory_buffer_urbanpy import Buffer

def UrbanPy_continual_train():
    urbanpy_args = get_args()
    urbanpy_args.model = 'UrbanPy'
    urbanpy_args.use_exf = True
    save_path = 'experiments/continual/{}-UrbanPy'.format(urbanpy_args.dataset)

    torch.manual_seed(urbanpy_args.seed)
    warnings.filterwarnings('ignore')
    print("mk dir {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def get_learning_rate(optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        return lr[-1]

    def load_init_model():
        load_path = 'experiments/single-task/{}-UrbanPy/best_epoch_P1.pt'.format(urbanpy_args.dataset)
        print("load from {}".format(load_path))
        model_state_dict = torch.load(load_path)["model_state_dict"]

        model = UrbanPy(in_channels=urbanpy_args.channels,
                    out_channels=urbanpy_args.channels,
                    img_width=urbanpy_args.from_reso,
                    img_height=urbanpy_args.from_reso,
                    n_residual_blocks=urbanpy_args.n_residuals,
                    base_channels=urbanpy_args.base_channels,
                    ext_dim=urbanpy_args.ext_dim,
                    ext_flag=urbanpy_args.ext_flag,
                    scales=urbanpy_args.scalers,
                    N=urbanpy_args.N,
                    n_res=urbanpy_args.n_res_propnet,
                    islocal=urbanpy_args.islocal,
                    compress=urbanpy_args.compress)
        model.load_state_dict(model_state_dict)
        if cuda:
            model = model.cuda()
        return model

    def load_best_model(task):
        load_path = '{}/best_epoch_{}.pt'.format(save_path, task)
        print("load from {}".format(load_path))
        model_state_dict = torch.load(load_path)["model_state_dict"]

        model = UrbanPy(in_channels=urbanpy_args.channels,
                    out_channels=urbanpy_args.channels,
                    img_width=urbanpy_args.from_reso,
                    img_height=urbanpy_args.from_reso,
                    n_residual_blocks=urbanpy_args.n_residuals,
                    base_channels=urbanpy_args.base_channels,
                    ext_dim=urbanpy_args.ext_dim,
                    ext_flag=urbanpy_args.ext_flag,
                    scales=urbanpy_args.scalers,
                    N=urbanpy_args.N,
                    n_res=urbanpy_args.n_res_propnet,
                    islocal=urbanpy_args.islocal,
                    compress=urbanpy_args.compress)
        model.load_state_dict(model_state_dict)
        if cuda:
            model = model.cuda()
        return model

    total_datapath = 'datasets'
    train_sequence = ["P1", "P2", "P3", "P4"]
    total_mses = {"P1":[np.inf], "P2":[np.inf], "P3":[np.inf], "P4":[np.inf]}
    best_epoch = {"P1":0, "P2":0, "P3":0, "P4":0}

    if urbanpy_args.initial_train:
        task_id = 0
    else:
        task_id = 1

    start_time = time.time()
    for task in train_sequence[task_id:]:
        print('==============Start to train {}=============='.format(task)) 
        task_id += 1

        train_dataloader = get_lapprob_dataloader(
                datapath= total_datapath, urbanpy_args= urbanpy_args, batch_size=urbanpy_args.batch_size, 
                mode='train', task_id = task_id)

        test_ds = get_lapprob_dataloader(
                datapath= total_datapath, urbanpy_args= urbanpy_args, batch_size=urbanpy_args.batch_size, 
                mode='test', task_id = task_id)

        if task_id == 1:
            base_model = UrbanPy(in_channels=urbanpy_args.channels,
                    out_channels=urbanpy_args.channels,
                    img_width=urbanpy_args.from_reso,
                    img_height=urbanpy_args.from_reso,
                    n_residual_blocks=urbanpy_args.n_residuals,
                    base_channels=urbanpy_args.base_channels,
                    ext_dim=urbanpy_args.ext_dim,
                    ext_flag=urbanpy_args.ext_flag,
                    scales=urbanpy_args.scalers,
                    N=urbanpy_args.N,
                    n_res=urbanpy_args.n_res_propnet,
                    islocal=urbanpy_args.islocal,
                    compress=urbanpy_args.compress).cuda()
            buffer = Buffer(urbanpy_args.buffer_size, urbanpy_args.n_tasks, urbanpy_args)
            continual_model = continual(base_model, buffer, urbanpy_args.n_tasks, urbanpy_args)
            print_model_parm_nums(base_model, urbanpy_args.model)

        if task_id > 1:
            if task_id == 2 and urbanpy_args.initial_train is not True:
                base_model = load_init_model()
                buffer = Buffer(urbanpy_args.buffer_size, urbanpy_args.n_tasks, urbanpy_args)
                buffer.seen_task = task_id - 1
                print('='*15,"load buffer of stage1 size:{}".format(urbanpy_args.buffer_size),'='*15)
                buffer.buffer_maps_ext = torch.load('stage1_maps_ext_{}.pt'.format(urbanpy_args.buffer_size))
                continual_model = continual(base_model, buffer, urbanpy_args.n_tasks, urbanpy_args)
            else:
                base_model = load_best_model(train_sequence[task_id-2])
                buffer.seen_task = task_id - 1
                continual_model = continual(base_model, buffer, urbanpy_args.n_tasks, urbanpy_args)
            print_model_parm_nums(base_model, urbanpy_args.model)

        for epoch in range(1, urbanpy_args.n_epochs+1):
            epoch_start_time = time.time()

            train_loss = 0
            # training phase
            continual_model.train_data_len = 0
            for i, flow_ext in enumerate(train_dataloader):
                train_loss += continual_model.observe(flow_ext, t= task_id)
            train_loss = train_loss / continual_model.train_data_len

            # validating phase
            base_model.eval()       
            if epoch % 5 == 0 or epoch == 1:
                for id in range(1, task_id+1):
                    mses = [0 for i in range(urbanpy_args.N)]
                    valid_dataloader = get_lapprob_dataloader(
                                        datapath= total_datapath, urbanpy_args= urbanpy_args, batch_size=32, 
                                        mode='valid', task_id = id)
                    for j, flow_ext_val in enumerate(valid_dataloader):
                        flows_v = flow_ext_val[:-1]; ext_v = flow_ext_val[-1]
                        densities, outs = base_model(flows_v[0], ext_v)   
                        preds = [out.cpu().detach().numpy() * urbanpy_args.scalers[j+1] for j, out in enumerate(outs)]
                        labels = [flow.cpu().detach().numpy() * urbanpy_args.scalers[j] for j, flow in enumerate(flows_v)] 
                        
                        for j, (pred, label) in enumerate(zip(preds, labels[1:])):
                            mses[j] += get_MSE(pred, label) * len(pred)   
                        mse = [mses / len(valid_dataloader.dataset) for mses in mses][-1]
                    if id == task_id and mse < np.min(total_mses[task]):
                        state = {'model_state_dict': base_model.state_dict(), 'epoch': epoch, 'task': task}
                        best_epoch[task] = epoch
                        torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))
                    total_mses[train_sequence[id-1]].append(mse)

                log = ('Task:{}|Epoch:{}|Loss:{:.3f}|Val_MSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}|Time_Cost:{:.2f}|Best_Epoch:{}'.format( 
                            task, epoch, train_loss, 
                            total_mses[train_sequence[0]][-1], total_mses[train_sequence[1]][-1], 
                            total_mses[train_sequence[2]][-1], total_mses[train_sequence[3]][-1], 
                            time.time() - epoch_start_time, best_epoch[task]))
                print(log)
                f = open('{}/train_process.txt'.format(save_path), 'a')
                f.write(log+'\n')
                f.close()
            else:
                mses = [0 for i in range(urbanpy_args.N)]
                valid_dataloader = get_lapprob_dataloader(
                                        datapath= total_datapath, urbanpy_args= urbanpy_args, batch_size=32, 
                                        mode='valid', task_id = task_id)
                for j, flow_ext_val in enumerate(valid_dataloader):
                    flows_v = flow_ext_val[:-1]; ext_v = flow_ext_val[-1]
                    densities, outs = base_model(flows_v[0], ext_v)   
                    preds = [out.cpu().detach().numpy() * urbanpy_args.scalers[j+1] for j, out in enumerate(outs)]
                    labels = [flow.cpu().detach().numpy() * urbanpy_args.scalers[j] for j, flow in enumerate(flows_v)]        
                    for j, (pred, label) in enumerate(zip(preds, labels[1:])):
                        mses[j] += get_MSE(pred, label) * len(pred)               
                    mse = [mses / len(valid_dataloader.dataset) for mses in mses][-1]

                if mse < np.min(total_mses[task]):
                    state = {'model_state_dict': base_model.state_dict(), 'epoch': epoch, 'task': task}
                    best_epoch[task] = epoch
                    torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))
                total_mses[train_sequence[id-1]].append(mse)

        model = load_best_model(task)
        model.eval()
        mses = [0 for i in range(urbanpy_args.N)]
        total_maes = [0 for i in range(urbanpy_args.N)]
        total_mapes = [0 for i in range(urbanpy_args.N)]
        for i, flow_ext in enumerate(test_ds):
            flows = flow_ext[:-1]; ext = flow_ext[-1]
            densities, outs = model(flows[0], ext)

            preds = [out.cpu().detach().numpy() * urbanpy_args.scalers[j+1] for j, out in enumerate(outs)]
            test_labels = [flow.cpu().detach().numpy() * urbanpy_args.scalers[j] for j, flow in enumerate(flows)] 

            for j, (pred, label) in enumerate(zip(preds, test_labels[1:])):
                mses[j] += get_MSE(pred, label) * len(pred)
                total_maes[j] += get_MAE(pred, label) * len(pred)
                total_mapes[j] += get_MAPE(pred, label) * len(pred)
        mse = [total_mse / len(test_ds.dataset) for total_mse in mses][-1]
        mae = [total_mae / len(test_ds.dataset) for total_mae in total_maes][-1]
        mape = [total_mre / len(test_ds.dataset) for total_mre in total_mapes][-1]

        log = ('{} test: MSE={:.6f}, MAE={:.6f}, MAPE={:.6f}'.format(task, mse, mae, mape))
        f = open('{}/test_results.txt'.format(save_path), 'a')
        f.write(log+'\n')
        f.close
        print(log)
        print('*' * 64)

    log = (
        f'Total running time: {(time.time()-start_time)//60:.0f}mins {(time.time()-start_time)%60:.0f}s')
    print(log)
    f = open('{}/test_results.txt'.format(save_path), 'a')
    f.write(log+'\n')
    f.close()
    print('*' * 64)
