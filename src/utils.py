import math
import torch
import numpy as np
import random
import pandas as pd
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
import os
import logging
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))

def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    # d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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



def earily_stop(val_acc_history, tasks, early_stop_step_single,
                early_stop_step_multi, required_progress):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    # TODO: add your code here
    if len(tasks) == 1:
        t = early_stop_step_single
    else:
        t = early_stop_step_multi

    if len(val_acc_history)>t:
        if val_acc_history[-1] - val_acc_history[-1-t] < required_progress:
            return True
    return False


def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))

    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)



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



class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        return torch.mean(squared_errors)
