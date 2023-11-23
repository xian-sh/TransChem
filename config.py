

args = {
    'model': 'TransChem',
    'save_dir': './results_right_ring1',
    'train_data_path': "./data/ring_train_2w.csv",
    'test_data_path': "./data/ring_test_2w.csv",
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


model_params = {
        'd_atom': 34,
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
