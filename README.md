# TransChem

**Paper**: TransChem: Data-based prediction of redox potentials via introducing chemical features into the transformer architecture

[paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01299)

## Abstract

Rapid and accurate prediction of basic physicochemical parameters of molecules will greatly accelerate the target-orientated design of novel reactions and materials but has been long challenging. Herein, a chemical language model-based deep learning method, TransChem, has been developed for the prediction of redox potentials of organic molecules. Embedding an effective molecular characterization (combining spatial and electronic features), a nonlinear molecular messaging approach (Mol-Attention), and a perturbation learning method, TransChem, shows high accuracy in predicting the redox potential of organic radicals comprising over 100,000 data (R2 > 0.97, MAE <0.09 V) and is generalized to the smaller 2,1,3-benzothiadiazole data set (<3000 data points) and electron affinity data set (660 data) with low MAE of 0.07 V and 0.18 eV, respectively. In this context, a self-developed data set, i.e., the oxidation potential (OP) of a full-space disubstituted phenol data set (OPP-data set, total set: 74,529), has been predicted by TransChem with a high-throughput, and active learning strategy. The rapid and reliable prediction of OP could hopefully accelerate the screening of plausible reagents in highly selective cross-coupling of phenol derivatives. This study presents an important attempt to guide language modeling with chemical knowledge, while TransChem demonstrates state-of-the-art (SOTA) predictive performance on redox potential prediction benchmark data sets for its better understanding of molecular design and conformational relationships.



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


| **Training Args** | **Value** | **Interpretation** |
| ---- |:-------------| :-------------| 
| model | TransChem | # model name |
| save_dir | ./results | # the folder path where the results are saved |
| train_data_path | ./data/ring_train_2w.csv | # file path for train data |
| test_data_path | ./data/ring_test_2w.csv | # file path for test data |
| pretained_path | None | # path to pre-trained model weights file | 
| train_num | all | # number of training samples |
| test_num | all | # number of testing samples | 
| pretrained | False | # whether to use pre-trained weights | 
| **noise** | True | # whether to add noise (or PNCGD) |
| noise_std | 0.2 | # standard deviation of noise |
| gpu | 0 | # GPU index to use |
| epochs | 230 | # number of training epochs |
| es_patience | 50 | # early stopping patience |
| lr | 0.0001 | # learning rate |
| step_size | 10 | # step size for learning rate scheduler |
| gamma | 0.96 | # leaining rate decay factor |
| batch_size | 64 | # batch size for data loaders |
| num_workers | 32 | # number of workers for data loading |
| seed | 42 | # random seed for reproducibility |
</br>


| **Model Args** | **Value** | **Interpretation** |
| ---- |:-------------| :-------------| 
| d_model | 1024 | # model dimensions of feedforward neural networks |
| N | 8 | # layers  of the model |
| h | 16 | # number of heads of attention module |
| N_dense | 1 | # layers of dense networks |
| trainable_lambda | False | # whether weights of attention and distance matrices etc. are trainable |
| lambda_attention | 0.5 | # the weights of the attention matrix |
| lambda_distance | 0. | # the weights of the distance matrix |
| leaky_relu_slope | 0.1 | # the parameter in LeakyRelu function |
| dense_output_nonlinearity | relu | # the nonlinearity operation in dense layers |
| distance_matrix_kernel | exp | # scaling functions in the distance matrix |
| dropout | 0.1 | # the dropout value |
| aggregation_type | mean | # the ggregation type of the resultes |
</br>


##  Citation

If you found this code useful please consider citing our [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01299):

    @article{doi:10.1021/acs.jcim.4c01299,
    author = {Si, Zhan and Liu, Deguang and Nie, Wan and Hu, Jingjing and Wang, Chen and Jiang, Tingting and Yu, Haizhu and Fu, Yao},
    title = {Data-Based Prediction of Redox Potentials via Introducing Chemical Features into the Transformer Architecture},
    journal = {Journal of Chemical Information and Modeling},
    volume = {64},
    number = {22},
    pages = {8453-8463},
    year = {2024},
    doi = {10.1021/acs.jcim.4c01299},
    note ={PMID: 39513760},
    URL = {https://doi.org/10.1021/acs.jcim.4c01299},
    eprint = {https://doi.org/10.1021/acs.jcim.4c01299}
    }


### LICENSE

The annotation files are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, see [./data/LICENSE](data/LICENSE). All the code are under [MIT](https://opensource.org/licenses/MIT) license, see [LICENSE](./LICENSE).

