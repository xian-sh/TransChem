"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def load_data_from_df(dataset_path, num, norm_lap=False, add_dummy_node=True, one_hot_formal_charge=False,
                      use_data_saving=True):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        norm_lap (bool): If True, the normalized Laplacian matrix will be used instead of the adjacency matrix.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    feat_stamp = f'{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}'
    feature_path = dataset_path.replace('.csv', f'{feat_stamp}.p')
    if use_data_saving and os.path.exists(feature_path):
        logging.info(f"Loading features stored at '{feature_path}'")
        x_all, y_all = pickle.load(open(feature_path, "rb"))

        if not isinstance(num, str):
            return x_all[:num], y_all[:num]
        elif num == 'all':
            return x_all, y_all
        else:
            raise 'num is int type or all'
            return

    data_df = pd.read_csv(dataset_path)

    if not isinstance(num, str):
        data_x = data_df.iloc[:num, 0].values
        data_y = data_df.iloc[:num, 1].values
    elif num == 'all':
        data_x = data_df.iloc[:, 0].values
        data_y = data_df.iloc[:, 1].values
    else:
        raise 'num is int type or all'

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all = load_data_from_smiles(data_x, data_y, norm_lap=norm_lap, add_dummy_node=add_dummy_node,
                                         one_hot_formal_charge=one_hot_formal_charge)
    if use_data_saving and not os.path.exists(feature_path):
        logging.info(f"Saving features at '{feature_path}'")
        pickle.dump((x_all, y_all), open(feature_path, "wb"))
        if not isinstance(num, str):
            return x_all[:num], y_all[:num]
        elif num == 'all':
            return x_all, y_all
        else:
            raise 'num is int type or all'

    return x_all, y_all


def load_data_from_smiles(x_smiles, labels, norm_lap, add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        norm_lap (bool): If True, the normalized Laplacian matrix will be used instead of the adjacency matrix.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all = [], []

    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist, sp3_mask = featurize_mol(mol, norm_lap, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist, sp3_mask])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all, y_all


def featurize_mol(mol, norm_lap, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        norm_lap (bool): If True, the normalized Laplacian matrix will be used instead of the adjacency matrix.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_feats = []
    sp3_labels = []
    for atom in mol.GetAtoms():
        feats, sp3 = get_atom_features(atom, one_hot_formal_charge)
        node_feats.append(feats)
        sp3_labels.append(sp3)
    node_features = np.array(node_feats)


    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        # 获取键的单双键类型，单键为1，双键为2，三键为3
        bond_type = bond.GetBondTypeAsDouble()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = bond_type

    sp3_mask = np.zeros_like(adj_matrix)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            # 如果两个原子之间有键，判断其键两端的原子是含有sp3杂化的原子,如果有，则标记该位置的sp3_mask为1
            if adj_matrix[i, j] > 0:
                if sp3_labels[i] == 1 or sp3_labels[j] == 1:
                    sp3_mask[i, j] = 1

    # 获取分子的度矩阵，即对角线元素为每个原子的度
    deg_matrix = np.diag(np.sum(adj_matrix, axis=1))

    # 计算拉普拉斯矩阵
    lap_matrix = deg_matrix - adj_matrix
    if norm_lap:
        # 计算归一化的拉普拉斯矩阵
        norm_lap_matrix = np.eye(mol.GetNumAtoms()) - np.dot(np.linalg.inv(deg_matrix), adj_matrix)
        lap_matrix = norm_lap_matrix

    # 计算距离矩阵
    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((lap_matrix.shape[0] + 1, lap_matrix.shape[1] + 1))
        m[1:, 1:] = lap_matrix
        lap_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, lap_matrix, dist_matrix, sp3_mask


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.  lens:11+6+5+3+5+3=33
    """
    attributes = []

    # 将原子的原子序数（列表中包含的原子，999为无效占位符）转换为one-hot向量
    # 5-B，6-C，7-N，8-O，9-F，15-P，16-S，17-Cl，35-Br，53-I
    # 11位
    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    # 每个原子的邻居数
    # 6位
    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )
    # 每个原子的总氢数
    # 5位
    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )
    # 每个原子的正式电荷
    # 3位
    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    # 每个原子的杂化类型
    # 5位
    attributes += one_hot_vector(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
    )
    sp3 = atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
    # 每个原子是否是自由基位点
    # 1位
    attributes.append(atom.GetNumRadicalElectrons() > 0)
    # 每个原子是否在环中
    # 1位
    attributes.append(atom.IsInRing())
    # 每个原子是否是芳香的
    # 1位
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32), sp3


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.sp3_mask = x[3]
        self.y = y
        self.index = index


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list, sp3_list = [], [], [], []
    labels = []
    index_list = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        sp3_list.append(pad_array(molecule.sp3_mask, (max_size, max_size)))
        index_list.append(molecule.index)

    adjacency_list = np.stack(adjacency_list)
    distance_list = np.stack(distance_list)
    features_list = np.stack(features_list)
    sp3_list = np.stack(sp3_list)
    labels = np.stack(labels)
    index_list = np.stack(index_list)

    out = [adjacency_list, features_list, distance_list, sp3_list, labels, index_list]
    out = [torch.from_numpy(features) for features in out]

    return out  #[FloatTensor(features) for features in (adjacency_list, features_list, distance_list, labels, index_list)]


def construct_dataset(x_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], i)
              for i, data in enumerate(zip(x_all, y_all))]
    return MolDataset(output)


def construct_loader(x, y, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
                                         collate_fn=mol_collate_func,
                                         shuffle=shuffle)
    return loader
