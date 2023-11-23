import pandas as pd
import torch
import numpy as np
import os

#settings-----------------------------------------------------   
root_dir = './data/ring'              # data root dir 
data_name = 'xxx.csv'                 # the name of the processing csv filw
save_train_name = 'xxx_train.csv'     
save_test_name = 'xxx_test.csv'

train_ratio = 0.8

seed = 0                              # randomly shuffle data 

#processing---------------------------------------------------
data_path = os.path.join(root_dir, data_name)
save_train_path = os.path.join(root_dir, save_train_name)
save_test_path = os.path.join(root_dir, save_test_name)

data = pd.read_csv(data_path)
data = data.sample(frac=1, random_state=seed)
data = data.reset_index(drop=True)

train_data = data.iloc[:int(len(data) * train_ratio), :]
test_data = data.iloc[int(len(data) * train_ratio):, :]

train_data.to_csv(save_train_path, index=False)
test_data.to_csv(save_test_path, index=False)
