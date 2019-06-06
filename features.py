import numpy as np


def distance(df):
    coord_0 = df.loc[:, ['x_0', 'y_0', 'z_0']].values
    coord_1 = df.loc[:, ['x_1', 'y_1', 'z_1']].values

    return np.linalg.norm(coord_1 - coord_0, axis=1)
