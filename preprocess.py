import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(center=False):
    train = pd.read_csv('train.csv', index_col='id')
    test = pd.read_csv('test.csv', index_col='id')
    coords = pd.read_csv('structures.csv')

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


def encode_labels(train, test):
    train, test = train.copy(), test.copy()
    le = LabelEncoder()

    for feature in ['type', 'atom_0', 'atom_1']:
        le.fit(train.loc[:, feature])
        train.loc[:, feature] = le.transform(train.loc[:, feature])
        test.loc[:, feature] = le.transform(test.loc[:, feature])

    return train, test


def add_atom_count(train, test, coords):
    counts_df = count_atoms(coords)

    return (df.merge(counts_df, how='left', on='molecule_name')
        for df in [train, test])


def add_distance(train, test):
    train_d, test_d = (distance(df) for df in [train, test])

    return (df.assign(d=d) 
        for df,d in zip([train, test], [train_d, test_d]))


def center_d(train, test):
    norms_0 = []
    norms_1 = []

    for df in [train, test]:
        xyz_0 = df.loc[:, ['x_0', 'y_0', 'z_0']].values
        xyz_1 = df.loc[:, ['x_1', 'y_1', 'z_1']].values

        norms_0.append(np.linalg.norm(xyz_0, axis=1))
        norms_1.append(np.linalg.norm(xyz_1, axis=1))

    return (df.assign(center_d_0=c0, center_d_1=c1)
        for df,c0,c1 in zip([train, test], norms_0, norms_1))


def center_cos(train, test):
    coss = []

    for df in [train, test]:
        xyz_0 = df.loc[:, ['x_0', 'y_0', 'z_0']].values
        xyz_1 = df.loc[:, ['x_1', 'y_1', 'z_1']].values

        dot = np.sum(xyz_0 * xyz_1, axis=1)
        norm_prod = (np.linalg.norm(xyz_0, axis=1) 
            * np.linalg.norm(xyz_1, axis=1))
        coss.append(dot / norm_prod)

    return (df.assign(center_cos=c) 
        for df,c in zip([train, test], coss))


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


def distance(df):
    coord_0 = df.loc[:, ['x_0', 'y_0', 'z_0']].values
    coord_1 = df.loc[:, ['x_1', 'y_1', 'z_1']].values

    return np.linalg.norm(coord_1 - coord_0, axis=1)