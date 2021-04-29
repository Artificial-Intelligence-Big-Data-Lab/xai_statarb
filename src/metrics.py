import numpy as np
import pandas as pd


def computecumsum(series):
    xspredsimple = np.array(series.values.flatten().tolist())
    mean_return_by_day = np.array(series.values.flatten().tolist()).mean()
    xspred = xspredsimple.cumsum()
    ipred = np.argmax(np.maximum.accumulate(xspred) - xspred)
    if ipred == 0:
        jpred = 0
    else:
        jpred = np.argmax(xspred[:ipred])

    mddpred = xspred[jpred] - xspred[ipred]

    computed_return = sum(series.values.flatten().tolist())
    romad = sum(series.values.flatten().tolist()) / mddpred

    return mddpred, computed_return, romad, xspred, ipred, jpred, mean_return_by_day


def rmse(y, p):
    r"""Root Mean Square Error.
    .. math::
        RMSE(\mathbf{y}, \mathbf{p}) = \sqrt{MSE(\mathbf{y}, \mathbf{p})},
    with
    .. math::
        MSE(\mathbf{y}, \mathbf{p}) = |S| \sum_{i \in S} (y_i - p_i)^2
    Parameters
    ----------
    y : array-like of shape [n_samples, ]
        ground truth.
    p : array-like of shape [n_samples, ]
        predicted labels.
    Returns
    -------
    z: float
        root mean squared error.
    """
    z = y - p
    return np.sqrt(np.mean(np.multiply(z, z)))


def mda(y_cr_test: pd.DataFrame):
    """ Mean Directional Accuracy """

    x = np.sign(y_cr_test['label'] - y_cr_test['label'].shift(1)) == np.sign(
        y_cr_test['predicted'] - y_cr_test['label'].shift(1))
    return np.count_nonzero(x.values.astype('int')) / len(x[~x.isnull()])


def hit_count(y_cr_test):
    x = (np.sign(y_cr_test['predicted']) == np.sign(y_cr_test['label'])).astype(int)
    return np.count_nonzero(x.values), np.count_nonzero(x.values) / len(x)
