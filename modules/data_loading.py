import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('data/train.csv', index_col='id')
    test = pd.read_csv('data/test.csv', index_col='id')
    coords = pd.read_csv('data/structures.csv')
    atoms = pd.read_csv('data/atoms.csv')
    bonds = pd.read_csv('data/bonds.csv')

    train = merge_base_data(train, atoms, bonds)
    test = add_coords(test, atoms, bonds)

    return train, test, coords


def add_coords(atom_pairs_df, coords_df):
    atom_pairs_df = pd.merge(atom_pairs_df, coords_df, how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'])
    atom_pairs_df = pd.merge(atom_pairs_df, coords_df, how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'],
        suffixes=('_0', '_1'), copy=False)
    atom_pairs_df.drop(columns=['atom_index_0', 'atom_index_1'],
        inplace=True)
    
    return atom_pairs_df


def merge_base_data(base_df, atoms_df, bonds_df):
    data_df = pd.merge(base_df, atoms_df, how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule', 'atom_index'])
    data_df = pd.merge(data_df, atoms_df, how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule', 'atom_index'],
        suffixes=('_0', '_1'))
    data_df = pd.merge(data_df, bonds_df, how='left',
        left_on=['molecule_name', 'atom_index_0', 'atom_index_1'],
        right_on=['molecule', 'atom_0', 'atom_1'])
    data_df.drop(columns=['bond_index', 'atom_index_0', 'atom_index_1'], 
        inplace=True)
