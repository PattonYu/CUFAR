import time
import torch
from model.modules.AKR import continual
from model.modules.memory_buffer import Buffer
from src.metrics import get_MSE, get_MAE, get_MAPE
from src.utils import get_dataloader, print_model_parm_nums
from src.args import get_args
from model.UrbanFM import UrbanFM
from model.FODE import FODE
from model.UrbanODE import UrbanODE
from model.CUFAR import CUFAR
from model.DeepLGR import DeepLGR
from src.urbanpy_train_continual import UrbanPy_continual_train
import numpy as np
import os

args = get_args()
if args.model == 'UrbanPy':
    UrbanPy_continual_train()
else:
    device = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    save_path = 'experiments/continual/{}-{}-{}-{}-{}'.format(
                                                    args.model,
                                                    args.dataset,
                                                    args.n_channels,
                                                    args.buffer_size,
                                                    args.minibatch_size)
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
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
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
        load_path = 'experiments/continual/CUFAR-TaxiBJ-128-1000-2-5/best_epoch_P1.pt'                       
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

    if args.initial_train:
        task_id = 0
    else:
        task_id = 1

    start_time = time.time()
    for task in train_sequence[task_id:]:
        print('='*15,'Start to train {}'.format(task),'='*15) 
        task_id += 1
        # load dataset
        train_ds = get_dataloader(args,
                    datapath= total_datapath, dataset= args.dataset, 
                    batch_size= args.batch_size, mode= 'train', task_id= task_id)

        test_ds = get_dataloader(args,
                    datapath= total_datapath, dataset= args.dataset,  
                    batch_size= 32, mode= 'test', task_id= task_id)
        # load model
        if task_id == 1:
            base_model = choose_model()
            if cuda:
                base_model = base_model.cuda()
            buffer = Buffer(args.buffer_size, args.n_tasks, args)
            continual_model = continual(base_model, buffer, args.n_tasks, args)
        else:
            if task_id == 2 and args.initial_train is not True:
                base_model = load_initial_model()
                buffer = Buffer(args.buffer_size, args.n_tasks, args)
                buffer.seen_task = task_id - 1
                print('='*15,"load buffer of stage1 size:{}".format(args.buffer_size), '='*15)
                buffer.buffer_c_maps = torch.load('buffers/{}_stage1_c_map_{}.pt'.format(args.dataset, args.buffer_size))
                buffer.buffer_f_maps = torch.load('buffers/{}_stage1_f_map_{}.pt'.format(args.dataset, args.buffer_size))
                buffer.buffer_ext = torch.load('buffers/{}_stage1_ext_{}.pt'.format(args.dataset, args.buffer_size))
                continual_model = continual(base_model, buffer, args.n_tasks, args)
            else:
                base_model = load_best_model(train_sequence[task_id-2])
                buffer.seen_task = task_id - 1
                continual_model = continual(base_model, buffer, args.n_tasks, args)

        print_model_parm_nums(continual_model.model, args.model)

        for epoch in range(1, args.n_epochs+1):
            epoch_start_time = time.time()
            train_loss = 0
            # training phase
            continual_model.train_data_len = 0
            for i, (c_map, f_map, exf) in enumerate(train_ds):
                train_loss += continual_model.observe(c_maps= c_map, f_maps= f_map, ext= exf, t= task_id)
            train_loss = train_loss / continual_model.train_data_len

            # validating phase, validate preivous tasks every 5 epochs.
            base_model.eval()
            if epoch % 5 == 0 or epoch == 1:
                for id in range(1, task_id+1):
                    val_mse, mse = 0, 0
                    valid_task = get_dataloader(args,
                                            datapath= total_datapath, dataset= args.dataset,  
                                            batch_size= 32, mode= 'valid', task_id= id)
                    for j, (c_map, f_map, exf) in enumerate(valid_task):
                        pred_f_map = base_model(c_map, exf)
                        pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
                        real = f_map.cpu().detach().numpy() * args.scaler_Y
                        mse += get_MSE(pred=pred, real=real) * len(c_map)                   
                    val_mse = mse / len(valid_task.dataset)
        
                    if id == task_id and val_mse < np.min(total_mses[task]):
                        state = {'model_state_dict': base_model.state_dict(), 'epoch': epoch, 'task': task}
                        best_epoch[task] = epoch
                        torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))

                    total_mses[train_sequence[id-1]].append(val_mse)
                log = ('Task:{}|Epoch:{}|Loss:{:.3f}|Val_MSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}|Time_Cost:{:.2f}|Best_Epoch:{}|lr:{}'.format( 
                                task, epoch, train_loss, 
                                total_mses[train_sequence[0]][-1], total_mses[train_sequence[1]][-1], 
                                total_mses[train_sequence[2]][-1], total_mses[train_sequence[3]][-1], 
                                time.time() - epoch_start_time, best_epoch[task], 
                                get_learning_rate(continual_model.optimizer)))
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
                    pred_f_map = base_model(c_map, exf)
                    pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
                    real = f_map.cpu().detach().numpy() * args.scaler_Y
                    mse += get_MSE(pred=pred, real=real) * len(c_map)
                    
                val_mse = mse / len(valid_task.dataset)

                if val_mse < np.min(total_mses[task]):
                    state = {'model_state_dict': base_model.state_dict(), 'epoch': epoch, 'task': task}
                    best_epoch[task] = epoch
                    torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))

                total_mses[task].append(val_mse)

                # log = ('Task:{}|Epoch:{}|Loss:{:.3f}|Val_MSE\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}|Time_Cost:{:.2f}|Best_Epoch:{}|lr:{}'.format( 
                #                 task, epoch, train_loss, 
                #                 total_mses[train_sequence[0]][-1], total_mses[train_sequence[1]][-1], 
                #                 total_mses[train_sequence[2]][-1], total_mses[train_sequence[3]][-1], 
                #                 time.time() - epoch_start_time, best_epoch[task], 
                #                 get_learning_rate(continual_model.optimizer)))
                # print(log)
                # f = open('{}/train_process.txt'.format(save_path), 'a')
                # f.write(log+'\n')
                # f.close()

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
