import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('train.csv', index_col='id')
    test = pd.read_csv('test.csv', index_col='id')
    coords = pd.read_csv('structures.csv')

    train = add_coords(train, coords)
    test = add_coords(test, coords)

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

