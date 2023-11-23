from __future__ import division
from __future__ import print_function

import time
import numpy as np
import random
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import copy
from sklearn.metrics import r2_score
import os
import logging
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from src.transformer import make_model
from src.featurization.data_utils import load_data_from_df, construct_loader

def setup_logger(name, save_dir, filename="training_log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False

    return logger

def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    # d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        return torch.mean(squared_errors)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

model_names = {
    1: 'MAT',
}

args = {
    'model': model_names[1],                                                
    'save_dir': './results_right_ring1',
    'train_data_path': r"C:\Users\Lenovo\PycharmProjects\Trans_R\data\newdataring\ring_ox_train.csv",
    'test_data_path': r"C:\Users\Lenovo\PycharmProjects\Trans_R\data\newdataring\ring_ox_test.csv",
    'pretained_path': "/gpfs/home/C22201088/ssz/Trans_R/results_new_translongnew/model.pt",   

    'train_num': 'all',            # input 1000 or 'all'
    'test_num': 'all',
    'pretrained':False,                                       
    'noise': True,                                         
    'noise_std': 0.0,
    'gpu': 0,                                                 
    'epochs': 230,
    'es_patience': 50,
    'lr': 0.0001,                                               
    'step_size': 10,                                            
    'gamma': 0.96,                                              
    'batch_size': 64,                                          
    'num_workers': 32,                                       
    'seed': 42,                                                
}

set_seed(args['seed'])
model_path = os.path.join(args['save_dir'], 'model.pt')
result_path = os.path.join(args['save_dir'], 'results.txt')
loss_path = os.path.join(args['save_dir'], 'train_loss.csv')
loss_img = os.path.join(args['save_dir'], 'loss.png')
pred_path = os.path.join(args['save_dir'], 'predict.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args['gpu'])


def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def test(model, loader):
    model.eval()
    model.cuda()
    index_ls = []
    pred_ls = []
    y_ls = []
    for batch in loader:
        with torch.no_grad():
            adjacency_matrix, node_features, distance_matrix, sp3_mask, y, index = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            node_features = node_features.cuda()
            batch_mask = batch_mask.cuda()
            adjacency_matrix = adjacency_matrix.cuda()
            distance_matrix = distance_matrix.cuda()
            pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

            pred_ls.append(pred)
            y_ls.append(y)
            index_ls.append(index)

    pred = torch.cat(pred_ls, dim=0).reshape(-1).cuda()  # (batch_size, 1)
    y = torch.cat(y_ls, dim=0).reshape(-1).cuda()
    index = torch.cat(index_ls, dim=0).reshape(-1).cuda()

    mae = F.l1_loss(pred.reshape(-1), y.reshape(-1))
    test_loss = mae
    rmse = torch.sqrt(test_loss)

    return test_loss, mae, rmse, y, pred, index


if __name__ == '__main__':

    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    logger = setup_logger(f"{args['model']}", args['save_dir'])
    logger.info('-' * 60)

    logger.info('The args are:\n' + f'{dict_to_markdown(args, max_str_len=120)}')

    train_X, train_y = load_data_from_df(args['train_data_path'], num=args['train_num'], one_hot_formal_charge=True)
    train_loader = construct_loader(train_X, train_y, args['batch_size'])
    train_d_atom = train_X[0][0].shape[1]  # It depends on the used featurization.

    test_X, test_y = load_data_from_df(args['test_data_path'], num=args['test_num'], one_hot_formal_charge=True)
    test_loader = construct_loader(test_X, test_y, args['batch_size'])

    logger.info(f'The train dataset contains {len(train_X)} samples.')
    logger.info(f'The test dataset contains {len(test_X)} samples.')

    model_params = {
        'd_atom': train_d_atom,
        'd_model': 1024,
        'N': 8,
        'h': 16,
        'N_dense': 1,
        'trainable_lambda': False,
        'lambda_attention': 0.5,
        'lambda_distance': 0.,
        'leaky_relu_slope': 0.1,
        'dense_output_nonlinearity': 'relu',
        'distance_matrix_kernel': 'exp',
        'dropout': 0.1,
        'aggregation_type': 'mean'
    }

    logger.info('The model params are:\n' + f'{dict_to_markdown(model_params, max_str_len=120)}')

    model = make_model(**model_params)
    if args['pretrained']:
        pretrained_name = args['pretained_path']  # This file should be downloaded first (See README.md).
        pretrained_state_dict = torch.load(pretrained_name)
        logger.info("loading pretrained weights from {}".format(pretrained_name))
        model_state_dict = model.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params /= 1e6
    logger.info('Total number of trainable parameters: {:.2f}M'.format(total_params))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args['lr'])  # , weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    start_time = time.time()

    logger.info('-' * 40)
    history = {'train_loss': [], 'val_loss': []}

    best_val_loss = 1e9
    best_model = model
    es = 0
    model.cuda()
    logger.info('Start training...')
    for epoch in range(args['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        train_epoch_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            adjacency_matrix, node_features, distance_matrix, sp3_mask, y, index = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            node_features = node_features.cuda()
            batch_mask = batch_mask.cuda()
            adjacency_matrix = adjacency_matrix.cuda()
            distance_matrix = distance_matrix.cuda()
            sp3_mask = sp3_mask.cuda()

            if args['noise']:
                torch.manual_seed(epoch)  # 设置PyTorch随机种子
                noise = torch.randn_like(distance_matrix) * args['noise_std'] * sp3_mask
                noise = noise.cuda()
                distance_matrix = distance_matrix + noise

            pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

            label = y.to(torch.float32).cuda()

            loss = F.l1_loss(pred.squeeze(), label.squeeze())
            # loss = rmse(pred.squeeze(), label.squeeze())
            train_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_epoch_loss /= len(train_loader)

        # ---validation---
        val_epoch_loss, val_epoch_mae, val_epoch_rmse, _, _, _ = test(model, test_loader)
        model.train()
        scheduler.step()

        # record epoch_loss
        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(val_epoch_loss)

        # print training process
        log = 'Epoch: {:03d}/{:03d}; ' \
              'AVG Training Loss (MAE):{:.8f}; ' \
              'AVG Val Loss (MAE):{:.8f};' \
              'lr:{:8f}'
        logger.info(log.format(epoch + 1, args['epochs'], train_epoch_loss, val_epoch_loss, current_lr))

        if current_lr != optimizer.param_groups[0]['lr']:
            logger.info('lr has been updated from {:.8f} to {:.8f}'.format(current_lr,
                                                                           optimizer.param_groups[0]['lr']))

        # determine whether stop early by val_epoch_loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = copy.deepcopy(model)
            es = 0
        else:
            es += 1
            logger.info("Counter {} of patience {}".format(es, args['es_patience']))
            if es > args['es_patience']:
                logger.info("Early stopping with best_val_loss {:.8f}".format(best_val_loss))
                break

    # best_model =  torch.load(model_path)
    end_time = time.time()
    logger.info("training time: {:.2f} min".format((end_time - start_time) / 60))

    torch.save(best_model.state_dict(), model_path)
    mse, mae, rmse, y, pred, smile = test(best_model, test_loader)
    r_2 = r2_score(y.cpu().numpy(), pred.cpu().numpy())
    ratio_02 = (torch.abs(y - pred) <= 0.2).sum() / y.size(0)
    ratio_01 = (torch.abs(y - pred) <= 0.1).sum() / y.size(0)

    logger.info("test result:\n"
                "MAE: {mae:.8f}\n"
                "RMSE: {rmse:.8f}\n"
                "R_2: {r_2:.5f}\n"
                "Ratio_02: {ratio_02:.5f}\n"
                "Ratio_01: {ratio_01:.5f}\n".format(mae=mae, rmse=rmse, r_2=r_2, ratio_02=ratio_02, ratio_01=ratio_01)
                )

    df = pd.DataFrame(history['train_loss'])
    df.to_csv(loss_path, index=False)

    smiles_ls = []
    y_ls = []
    pred_ls = []
    smiles_ls.extend(smile.cpu().tolist())
    y_ls.extend(y.cpu().tolist())
    pred_ls.extend(pred.cpu().tolist())

    pred_data = {
        'smiles': smiles_ls,
        'y': y_ls,
        'pred': pred_ls,
    }
    pred_df = pd.DataFrame(pred_data)
    pred_df['pred'] = pred_df['pred'].apply(lambda x: round(x, 2))
    pred_df.to_csv(pred_path, index=False)

    loss_data = pd.read_csv(loss_path)
    plt.plot(loss_data)
    plt.savefig(loss_img)
    plt.show()



