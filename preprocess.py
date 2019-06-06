import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(center=False):
    train = pd.read_csv('train.csv', index_col='id')
    test = pd.read_csv('test.csv', index_col='id')
    coords = pd.read_csv('structures.csv')

    if center:
        coords = center_coords(coords)

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


def center_coords(df):
    mu_df = (df.groupby(['molecule_name'])[['x', 'y', 'z']].sum() 
             / df.groupby(['molecule_name'])[['x', 'y', 'z']].count())
    mu_df = (mu_df
             .rename(columns={'x': 'x_mu', 'y': 'y_mu', 'z': 'z_mu'})
             .reset_index())
    
    df = (df
          .merge(mu_df, how='left', on=['molecule_name'])
          .drop(columns=['x_mu', 'y_mu', 'z_mu']))
    
    return df


def encode_labels(df):
    df = df.copy()
    enc = LabelEncoder()

    for feature in ['type', 'atom_0', 'atom_1']:
        enc.fit(df.loc[:, feature])
        df.loc[:, feature] = enc.transform(df.loc[:, feature])

    df.drop(columns=['molecule_name'], inplace=True)

    return df


def count_atoms(coords_df):
    return (pd
            .get_dummies(coords_df, columns=['atom'])
            .groupby('molecule_name')
            .sum()
            .drop(columns=['atom_index', 'x', 'y', 'z'])
            .reset_index())


def split_validation(train_df, test_size = .2):
    y = train_df.loc[:, 'scalar_coupling_constant'].copy()
    x = train_df.drop(columns=['scalar_coupling_constant'])

    x_train, x_val, y_train, y_val = \
        train_test_split(x, y, test_size=test_size, random_state=0)

    return x_train, x_val, y_train, y_val
