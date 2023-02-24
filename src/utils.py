import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_dataloader(args, datapath, dataset= "TaxiBJ", batch_size= 16, mode='train', task_id=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X = None
    Y = None
    ext = None

    sequence = ['P1', 'P2', 'P3', 'P4']

    ori_datapath = os.path.join(datapath, dataset)
    if mode == 'train':
        shuffle= True
        task = sequence[task_id-1]
        print("# load {} datset {}".format(mode, task))
        datapath = os.path.join(ori_datapath, task)
        datapath = os.path.join(datapath, mode)
        # if X is None:
        if dataset == "TaxiBJ":
            X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
            Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
            ext = np.load(os.path.join(datapath, 'ext.npy'))
        elif dataset == "TaxiNYC":
            X = np.load(os.path.join(datapath, 'X_16.npy')) / args.scaler_X
            Y = np.load(os.path.join(datapath, 'X_64.npy')) / args.scaler_Y
            ext = np.load(os.path.join(datapath, 'ext.npy'))

    else:
        shuffle= False        
        task = sequence[task_id-1]
        if mode == 'test':
            print("# load {} datset {}".format(mode, task))
        datapath = os.path.join(ori_datapath, task)
        datapath = os.path.join(datapath, mode)
        if dataset == "TaxiBJ":
            X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
            Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
            ext = np.load(os.path.join(datapath, 'ext.npy'))
        elif dataset == "TaxiNYC":
            X = np.load(os.path.join(datapath, 'X_16.npy')) / args.scaler_X
            Y = np.load(os.path.join(datapath, 'X_64.npy')) / args.scaler_Y
            ext = np.load(os.path.join(datapath, 'ext.npy'))

    X = Tensor(np.expand_dims(X, 1))
    Y = Tensor(np.expand_dims(Y, 1))
    ext = Tensor(ext)

    assert len(X) == len(Y)
    if mode != 'valid':
        print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, Y, ext)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle= shuffle)
    return dataloader


def get_dataloader_joint(args, datapath, dataset= "TaxiBJ", batch_size= 16, mode='train', task_id=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X = None
    Y = None
    ext = None
    sequence = ['P1', 'P2', 'P3', 'P4']

    ori_datapath = os.path.join(datapath, dataset)
    if mode == 'train':
        shuffle = True    
        for task in sequence[:task_id]:
            if task != sequence[task_id-1]:
                for task_mode in ['train', 'valid', 'test']:
                    print("# load {} datset {}".format(task_mode, task))
                    datapath = os.path.join(ori_datapath, task)
                    datapath = os.path.join(datapath, task_mode)
                    if X is None:
                        if dataset == "TaxiBJ":
                            X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
                            Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
                            ext = np.load(os.path.join(datapath, 'ext.npy'))
                        elif dataset == "TaxiNYC":
                            X = np.load(os.path.join(datapath, 'X_16.npy')) / args.scaler_X
                            Y = np.load(os.path.join(datapath, 'X_64.npy')) / args.scaler_Y
                            ext = np.load(os.path.join(datapath, 'ext.npy'))
                    else:
                        if dataset == "TaxiBJ":
                            X = np.concatenate([X, np.load(os.path.join(datapath, 'X.npy'))], axis= 0) / args.scaler_X
                            Y = np.concatenate([Y, np.load(os.path.join(datapath, 'Y.npy'))], axis= 0) / args.scaler_X
                            ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis= 0)
                        elif dataset == "TaxiNYC":
                            X = np.concatenate([X, np.load(os.path.join(datapath, 'X_16.npy'))], axis= 0) / args.scaler_X
                            Y = np.concatenate([Y, np.load(os.path.join(datapath, 'X_64.npy'))], axis= 0) / args.scaler_X
                            ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis= 0)

            else:
                print("# load {} datset {}".format(mode, task))
                datapath = os.path.join(ori_datapath, task)
                datapath = os.path.join(datapath, mode)
                if X is None:
                    if dataset == "TaxiBJ":
                        X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
                        Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
                        ext = np.load(os.path.join(datapath, 'ext.npy'))
                    elif dataset == "TaxiNYC":
                        X = np.load(os.path.join(datapath, 'X_16.npy')) / args.scaler_X
                        Y = np.load(os.path.join(datapath, 'X_64.npy')) / args.scaler_Y
                        ext = np.load(os.path.join(datapath, 'ext.npy'))
                else:
                    if dataset == "TaxiBJ":
                        X = np.concatenate([X, np.load(os.path.join(datapath, 'X.npy'))], axis= 0) / args.scaler_X
                        Y = np.concatenate([Y, np.load(os.path.join(datapath, 'Y.npy'))], axis= 0) / args.scaler_X
                        ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis= 0)
                    elif dataset == "TaxiNYC":
                        X = np.concatenate([X, np.load(os.path.join(datapath, 'X_16.npy'))], axis= 0) / args.scaler_X
                        Y = np.concatenate([Y, np.load(os.path.join(datapath, 'X_64.npy'))], axis= 0) / args.scaler_X
                        ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis= 0)
    else:
        shuffle = False
        task = sequence[task_id-1]
        print("# load {} datset {}".format(mode, task))
        datapath = os.path.join(ori_datapath, task)
        datapath = os.path.join(datapath, mode)
        if X is None:
            if dataset == "TaxiBJ":
                X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
                Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
                ext = np.load(os.path.join(datapath, 'ext.npy'))
            elif dataset == "TaxiNYC":
                X = np.load(os.path.join(datapath, 'X_16.npy')) / args.scaler_X
                Y = np.load(os.path.join(datapath, 'X_64.npy')) / args.scaler_Y
                ext = np.load(os.path.join(datapath, 'ext.npy'))
        else:
            if dataset == "TaxiBJ":
                X = np.concatenate([X, np.load(os.path.join(datapath, 'X.npy'))], axis= 0) / args.scaler_X
                Y = np.concatenate([Y, np.load(os.path.join(datapath, 'Y.npy'))], axis= 0) / args.scaler_X
                ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis= 0)
            elif dataset == "TaxiNYC":
                X = np.concatenate([X, np.load(os.path.join(datapath, 'X_16.npy'))], axis= 0) / args.scaler_X
                Y = np.concatenate([Y, np.load(os.path.join(datapath, 'X_64.npy'))], axis= 0) / args.scaler_X
                ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis= 0)

    X = Tensor(np.expand_dims(X, 1))
    Y = Tensor(np.expand_dims(Y, 1))
    ext = Tensor(ext)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))
    data = torch.utils.data.TensorDataset(X, Y, ext)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle= shuffle)
    return dataloader


def get_gt_densities(flows, opt):
    inp = flows[0] * opt.scaler_dict[opt.scales[0]]
    scale0 = opt.scales[0]
    out, masks = [], []

    for i, f in enumerate(flows[1:]):
        scale_ = opt.scales[i+1]
        inp_ = F.upsample(inp, scale_factor=scale_//scale0)
        masks.append(inp_ != 0)
        f0 = inp_ + 1e-9
        f_ = f*opt.scaler_dict[opt.scales[i+1]]
        out.append(f_/f0)
    return out, masks


def get_lapprob_dataloader(datapath, args, batch_size=2, mode='train', task_id = 0):
    sequence = ['P1', 'P2', 'P3', 'P4']
    task = sequence[task_id-1]
    datapath = os.path.join(datapath, args.dataset)
    datapath = os.path.join(datapath, task)
    datapath = os.path.join(datapath, mode)
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Xs = list()

    for scale in args.scales:
        if scale == 16:
            Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X_%d.npy'%scale)), 1)) / args.scaler_dict[scale])
        elif scale == 32:
            Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1)) / args.scaler_dict[scale])
        elif scale == 64:
            Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X_%d.npy'%scale)), 1)) / args.scaler_dict[scale])
        elif scale == 128:
            Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1)) / args.scaler_dict[scale])
    ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))

    Xs.append(ext)
    
    data = torch.utils.data.TensorDataset(*Xs)
    for scale in args.scales:
        if mode == 'train':
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))
    return total_num
