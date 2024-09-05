# TransChem

**Paper**: TransChem: Data-based prediction of redox potentials via introducing chemical features into the transformer architecture

<div align="center">
  <img src="./assets/ToC.png" alt="Table of Contents" width="500" height="400">
</div>



## Approach

<div align="center">
  <img src="./assets/mian_model.png" alt="Approach" width="800" height="400">
</div>

## Usage

- [Getting Started](#getting-started)

  - [Prerequisites](#prerequisites)

  - [Prediction](#prediction)

  - [Train TransChem on your own dataset](#train-transchem-on-your-own-dataset)

- [Acknowledgement](#acknowledgement)

- [LICENSE](#license)

### Getting Started

#### Prerequisites

1.  Clone this repo

<!---->

    git clone https://github.com/xianHUJINGJING/TransChem.git
    cd example

1.  Prepare model weight

Download [model_xxx.pt](https://drive) (160M), put it under './model_weight' directory.

1.  Install dependencies.

This code requires Python 3.7, PyTorch, and a few other Python libraries.
We recommend creating conda environment and installing all the dependencies as follows:

    # create conda env
    conda create --name transchem python=3.7
    # activate env
    conda actiavte transchem
    # install pytorch with CUDA 11.0
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
    # install other python packages
    pip install pickle rdkit sklearn random numpy pandas

The PyTorch version we tested is `1.9.0`.



#### Prediction

    cd example

#### Train TransChem on your own dataset

To train TransChem on your own dataset, please prepare your dataset annotations following the format


Training can be launched by running the following command:

    python train.py

For more configurable options, please checkout our config file [config.py](config.py).


| **Training Args** | |
| ---- |:-------------:| 
| model | TransChem | 
| save_dir | ./results | 
| train_data_path | ./data/ring_train_2w.csv | 
| test_data_path | ./data/ring_test_2w.csv | 
| pretained_path | None | 
| train_num | all | 
| test_num | all | 
| pretrained | False | 
| **noise** | True | 
| noise_std | 0.0 | 
| gpu | 0 | 
| epochs | 230 | 
| es_patience | 50 | 
| lr | 0.0001 | 
| step_size | 10 | 
| gamma | 0.96 |
| batch_size | 64 |
| num_workers | 32 |
| seed | 42 |
</br>


| **Model Args** | |
| ---- |:-------------:| 
| d_model | 1024 | 
| N | 8 | 
| h | 16 | 
| N_dense | 1 | 
| trainable_lambda | False | 
| lambda_attention | 0.5 | 
| lambda_distance | 0. | 
| leaky_relu_slope | 0.1 | 
| dense_output_nonlinearity | relu | 
| distance_matrix_kernel | exp | 
| dropout | 0.1 | 
| aggregation_type | mean | 
</br>


### LICENSE

The annotation files are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, see [./data/LICENSE](data/LICENSE). All the code are under [MIT](https://opensource.org/licenses/MIT) license, see [LICENSE](./LICENSE).

