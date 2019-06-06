import numpy as np
import preprocess
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor


def eval_metric(y, y_hat, j_type, floor=1e-9):
    """
    From https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y - y_hat).abs().groupby(j_type).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def predict_hgbr(x_train, y_train, x_test):
    x_train, x_test = (preprocess.encode_labels(x) 
        for x in [x_train, x_test])

    hgbr = HistGradientBoostingRegressor()
    hgbr.fit(x_train, y_train)
    return hgbr.predict(x_test)
