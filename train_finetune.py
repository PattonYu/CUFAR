import os
device = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = device
import time
import torch
import torch.nn.functional as F
from src.metrics import get_MSE, get_MAE, get_MAPE
from src.utils import get_dataloader, print_model_parm_nums
from src.args import get_args
from src.urbanpy_train_finetune import UrbanPy_finetune_train
from model.UrbanFM import UrbanFM
from model.FODE import FODE
from model.UrbanODE import UrbanODE
from model.DeepLGR import DeepLGR
from model.CUFAR import CUFAR
import numpy as np

args = get_args()
if args.model == 'UrbanPy':
    UrbanPy_finetune_train()
else:
    save_path = 'experiments/fine-tune/{}-{}-{}'.format(
                                                args.model,
                                                args.dataset,
                                                args.n_channels)
    torch.manual_seed(args.seed)

    print("mk dir {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    print('device:', device)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def get_learning_rate(optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        return lr[-1]

    def choose_model():
        if args.model == 'CUFAR':
            model = CUFAR(height=args.height, width=args.width, use_exf=args.use_exf,
                        scale_factor=args.scale_factor, channels=args.n_channels, 
                        sub_region= args.sub_region, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args= args)
        elif args.model == 'UrbanFM':
            model = UrbanFM(in_channels=1, out_channels=1, n_residual_blocks=16,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        elif args.model == 'FODE':
            model = FODE(in_channels=1, out_channels=1, n_residual_blocks=16,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        elif args.model == 'UrbanODE':
            model = UrbanODE(in_channels=1, out_channels=1, n_residual_blocks=16,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        elif args.model == 'DeepLGR':
            model = DeepLGR(in_channels=1, out_channels=1, n_residual_blocks=12,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag=  args.use_exf, predictor='td',
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        return model

    def load_initial_model():
        load_path = 'experiments/single-task/{}-{}-{}'.format(
                                                    args.model,
                                                    args.dataset,
                                                    args.n_channels)                   
        print("load from {}".format(load_path))
        model_state_dict = torch.load(load_path)["model_state_dict"]
        model = choose_model()
        model.load_state_dict(model_state_dict)
        if cuda:
            model = model.cuda()
        return model

    def load_best_model(task):
        load_path = '{}/best_epoch_{}.pt'.format(save_path, task)
        print("load from {}".format(load_path))
        model_state_dict = torch.load(load_path)["model_state_dict"]
        model = choose_model()
        model.load_state_dict(model_state_dict)
        if cuda:
            model = model.cuda()
        return model

    total_datapath = 'datasets'
    train_sequence = ["P1", "P2", "P3", "P4"]
    total_mses = {"P1":[np.inf], "P2":[np.inf], "P3":[np.inf], "P4":[np.inf]}
    best_epoch = {"P1":0, "P2":0, "P3":0, "P4":0}

    start_time = time.time()

    if args.initial_train:
        task_id = 0
    else:
        task_id = 1

    for task in train_sequence[task_id:]:
        print('='*15,'Start to train {}'.format(task),'='*15)
        task_id += 1

        train_ds = get_dataloader(args,
                    datapath= total_datapath, dataset= args.dataset, 
                    batch_size= args.batch_size, mode= 'train', task_id= task_id)

        test_ds = get_dataloader(args,
                    datapath= total_datapath, dataset= args.dataset,  
                    batch_size= 32, mode= 'test', task_id= task_id)

        if task_id == 1:
            model = choose_model()
            if cuda:
                model = model.cuda()
        else:
            if task_id == 2 and args.initial_train is not True:
                model = load_initial_model()
            else:
                model = load_best_model(train_sequence[task_id-2])
        print_model_parm_nums(model, args.model)

        optimizer = torch.optim.Adam(
                    model.parameters(), lr= args.lr, betas=(args.b1, args.b1))
        criterion = F.mse_loss

        for epoch in range(1, args.n_epochs+1):
            epoch_start_time = time.time()
            train_loss = 0
            # training phase
            for i, (c_map, f_map, exf) in enumerate(train_ds):
                model.train()
                optimizer.zero_grad()

                pred_f_map = model(c_map, exf) * args.scaler_Y
                loss = criterion(pred_f_map, f_map * args.scaler_Y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(c_map)
            train_loss /= len(train_ds.dataset)

            # validating phase, validate preivous tasks every 5 epochs.
            model.eval()
            if epoch % 5 == 0 or epoch == 1:
                for id in range(1, task_id+1):
                    val_mse, mse = 0, 0
                    valid_task = get_dataloader(args,
                                            datapath= total_datapath, dataset= args.dataset,  
                                            batch_size= 32, mode= 'valid', task_id= id)
                    for j, (c_map, f_map, exf) in enumerate(valid_task):
                        pred_f_map = model(c_map, exf)
                        pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
                        real = f_map.cpu().detach().numpy() * args.scaler_Y
                        mse += get_MSE(pred=pred, real=real) * len(c_map)                   
                    val_mse = mse / len(valid_task.dataset)
        
                    if id == task_id and val_mse < np.min(total_mses[task]):
                        state = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'task': task}
                        best_epoch[task] = epoch
                        torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))

                    total_mses[train_sequence[id-1]].append(val_mse)
                # log print
                log = ('Task:{}|Epoch:{}|Loss:{:.3f}|Val_MSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}|Time_Cost:{:.2f}|Best_Epoch:{}|lr:{}'.format( 
                                task, epoch, train_loss, 
                                total_mses[train_sequence[0]][-1], total_mses[train_sequence[1]][-1], 
                                total_mses[train_sequence[2]][-1], total_mses[train_sequence[3]][-1], 
                                time.time() - epoch_start_time, best_epoch[task], 
                                get_learning_rate(optimizer)))
                print(log)
                f = open('{}/train_process.txt'.format(save_path), 'a')
                f.write(log+'\n')
                f.close()
            else:
                val_mse, mse = 0, 0
                valid_task = get_dataloader(args,
                                        datapath= total_datapath, dataset= args.dataset,  
                                        batch_size= 32, mode= 'valid', task_id= task_id)
                for j, (c_map, f_map, exf) in enumerate(valid_task):
                    pred_f_map = model(c_map, exf)
                    pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
                    real = f_map.cpu().detach().numpy() * args.scaler_Y
                    mse += get_MSE(pred=pred, real=real) * len(c_map)
                    
                val_mse = mse / len(valid_task.dataset)

                if val_mse < np.min(total_mses[task]):
                    state = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'task': task}
                    best_epoch[task] = epoch
                    torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))

                total_mses[task].append(val_mse)

        model = load_best_model(task)
        model.eval()

        total_mse, total_mae, total_mape = 0, 0, 0
        for i, (c_map, f_map, eif) in enumerate(test_ds):
            pred_f_map = model(c_map, eif)
            pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
            real = f_map.cpu().detach().numpy() * args.scaler_Y
            total_mse += get_MSE(pred=pred, real=real) * len(c_map)
            total_mae += get_MAE(pred=pred, real=real) * len(c_map)
            total_mape += get_MAPE(pred=pred, real=real) * len(c_map)
        mse = total_mse / len(test_ds.dataset)
        mae = total_mae / len(test_ds.dataset)
        mape = total_mape / len(test_ds.dataset)

        log = ('{} Training test: MSE={:.6f}, MAE={:.6f}, MAPE={:.6f}'.format(task, mse, mae, mape))
        f = open('{}/test_results.txt'.format(save_path), 'a')
        f.write(log+'\n')
        f.close()
        print(log)
        print('*' * 64)

    log = (
        f'Total running time: {(time.time()-start_time)//60:.0f}mins {(time.time()-start_time)%60:.0f}s')
    print(log)
    f = open('{}/test_results.txt'.format(save_path), 'a')
    f.write(log+'\n')
    f.close()
    print('*' * 64)
