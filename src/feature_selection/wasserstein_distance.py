import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d


def wasserstein_distance_from_samples(sample_p: np.ndarray,
                                      sample_q: np.ndarray) -> float:
    return continuous_wasserstein_distance_from_sample(sample_p=sample_p,
                                                       sample_q=sample_q)


def wasserstein_distance_from_kde(p: sm.nonparametric.KDEUnivariate,
                                  q: sm.nonparametric.KDEUnivariate) -> float:
    domain = set(p.support) | set(q.support)
    domain = sorted(domain)
    U = interp1d(p.support, p.cdf, kind='nearest', bounds_error=False, fill_value="extrapolate")
    V = interp1d(q.support, q.cdf, kind='nearest', bounds_error=False, fill_value="extrapolate")
    return sum((np.array(domain)[1:] - np.array(domain)[:-1]) * (np.abs(U(domain[1:]) - V(domain[1:]))))


def continuous_wasserstein_distance_from_sample(sample_p: np.ndarray,
                                                sample_q: np.ndarray) -> float:
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return wasserstein_distance_from_kde(kde_p, kde_q)
