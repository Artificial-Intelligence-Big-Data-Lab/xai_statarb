from scipy.stats import wasserstein_distance
import statsmodels.api as sm
import numpy as np


def wasserstein_distance_from_sample(sample_p: np.ndarray, sample_q: np.ndarray):
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()
    return wasserstein_distance(kde_p.density, kde_q.density)
