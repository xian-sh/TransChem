import pandas as pd
import torch
import numpy as np

data_path = r'C:\Users\Lenovo\PycharmProjects\MAT\data\op_high_scoring_redox_potentials.csv'

# 读取数据
data = pd.read_csv(data_path)

# 随机打乱数据
data = data.sample(frac=1, random_state=0)
data = data.reset_index(drop=True)

# 前80%为训练集，后20%为测试集，保存为csv文件
train_data = data.iloc[:int(len(data) * 0.8), :]
test_data = data.iloc[int(len(data) * 0.8):, :]
train_data.to_csv(r'C:\Users\Lenovo\PycharmProjects\MAT\data\train.csv', index=False)
test_data.to_csv(r'C:\Users\Lenovo\PycharmProjects\MAT\data\test.csv', index=False)