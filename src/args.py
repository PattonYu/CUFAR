import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default= 100,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, # for DeepLGR, lr= 0.005
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_channels', type=int, default=128,
                        help='number of channels')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of flow image channels')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--use_exf', action='store_true', default=True,
                        help='External influence factors')
    parser.add_argument('--height', type=int, default=32,
                        help='height of the input map')
    parser.add_argument('--width', type=int, default=32,
                        help='weight of the input map')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='upscaling factor')
    parser.add_argument('--model', type=str, default='CUFAR',
                        help='chose model to use', )
                        # choices=['UrbanFM', 'FODE', 'UrbanODE', 'DeepLGR', 'UrbanPy', 'CUFAR'])
    parser.add_argument('--scaler_X', type=int, default=1,
                        help='scaler of coarse-grained flows')
    parser.add_argument('--scaler_Y', type=int, default=1,
                        help='scaler of fine-grained flows')
    parser.add_argument('--c_map_shape', type= int, default= 32,
                        choices=[16, 32])
    parser.add_argument('--f_map_shape', type= int, default= 128,
                        choices=[64, 128])
    parser.add_argument('--ext_shape', type= int, default= 7)
    parser.add_argument('--dataset', type=str, default='TaxiBJ', choices= ['TaxiBJ', 'TaxiNYC'],
                        help='dataset name')
    parser.add_argument('--sub_region', type= int, default=4,
                        help= 'sub regions number H\' and W\' in the paper')
    parser.add_argument('--initial_train', action='store_true', default= False, 
                        help= 'chose to train frist stage')
    parser.add_argument('--buffer_size', type= int, default= 1000, 
                        help='size of the memory buffer M')
    parser.add_argument('--n_tasks', type= int, default= 4, 
                        help='number of tasks')
    parser.add_argument('--minibatch_size', type= int, default= 2,
                        help='size of the sub-memory buffer M_sub')
    
    ## UrbanPy parameters

    parser.add_argument('--n_residuals', type=str, default='16,16',
                        help='number of residual units')
    parser.add_argument('--n_res_propnet', type=int, default=1,
                        help='number of res_layer in proposal net')
    parser.add_argument('--from_reso', type=int, choices=[8, 16, 32, 64, 128], default= 32,
                        help='coarse-grained input resolution')
    parser.add_argument('--to_reso', type=int, choices=[16, 32, 64, 128], default= 128,
                        help='fine-grained input resolution')
    parser.add_argument('--islocal', action='store_true', default = True,
                        help='whether to use external factors')
    parser.add_argument('--loss_weights', type=str, default='1,1,1', 
                        help='weights on multiscale loss')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='alpha')
    parser.add_argument('--compress', type=int, default=2,
                        help='compress channels')
    parser.add_argument('--coef', type=float, default=0.5,
                        help='the weight of proposal net')
    parser.add_argument('--save_diff', default=False, action='store_true',
                        help='save differences'),

    opt = parser.parse_args()
    if opt.dataset == 'TaxiNYC':
        opt.width, opt.height = 16, 16
        opt.scaler_dict = {16:1, 32:1, 64:1}
        opt.from_reso = 16
        opt.to_reso = 64
    elif opt.dataset == 'TaxiBJ':
        opt.width, opt.height = 32, 32
        opt.scaler_dict = {32:1, 64:1, 128:1}
        opt.from_reso = 32
        opt.to_reso = 128
    opt.c_map_shape = opt.width
    opt.f_map_shape = opt.width * opt.scale_factor
                    # {32:1500, 64:300, 128:100}
    opt.n_residuals = [int(_) for _ in opt.n_residuals.split(',')]
    opt.loss_weights = [float(_) for _ in opt.loss_weights.split(',')]
    opt.N = int(np.log(opt.to_reso / opt.from_reso)/np.log(2))

    opt.n_residuals = opt.n_residuals[:opt.N]
    assert opt.from_reso < opt.to_reso, 'invalid resolution, from {} to {}'.format(opt.from_reso, opt.to_reso)
    opt.scales = [opt.from_reso*2**i for i in range(opt.N+1)]
    opt.scalers= [opt.scaler_dict[key] for key in opt.scales]
    return opt
