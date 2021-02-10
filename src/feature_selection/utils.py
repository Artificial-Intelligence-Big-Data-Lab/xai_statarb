import typing as tp

import numpy as np
import statsmodels.api as sm
from cubature import cubature


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert (d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))


def relative_entropy_from_samples(sample_p: np.ndarray,
                                  sample_q: np.ndarray,
                                  base: float = np.e,
                                  discrete: bool = False) -> float:
    return continuous_relative_entropy_from_sample(sample_p=sample_p,
                                                   sample_q=sample_q,
                                                   base=base)


def jensen_shannon_divergence_from_samples(sample_p: np.ndarray,
                                           sample_q: np.ndarray,
                                           base: float = np.e,
                                           discrete: bool = False) -> float:
    return continuous_jensen_shannon_divergence_from_sample(sample_p=sample_p,
                                                            sample_q=sample_q,
                                                            base=base)


def continuous_relative_entropy_from_sample(sample_p: np.ndarray,
                                            sample_q: np.ndarray,
                                            base: float = np.e,
                                            eps_abs: float = 1.49e-08,
                                            eps_rel: float = 1.49e-08) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p
                D_KL(p||q) = E_p [log(p/q)]
    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return relative_entropy_from_kde(p=kde_p,
                                     q=kde_q,
                                     base=base,
                                     eps_abs=eps_abs,
                                     eps_rel=eps_rel)


def relative_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                              q: sm.nonparametric.KDEUnivariate,
                              base: float = np.e,
                              eps_abs: float = 1.49e-08,
                              eps_rel: float = 1.49e-08) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p
                D_KL(p||q) E_p [log(p/q)]
    given by the statsmodels kde objects via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    if not _does_support_overlap(p, q):
        raise ValueError('The support of p and q does not overlap.')

    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))
    return relative_entropy_from_densities_with_support(p=p.evaluate,
                                                        q=q.evaluate,
                                                        a=a,
                                                        b=b,
                                                        base=base,
                                                        eps_abs=eps_abs,
                                                        eps_rel=eps_rel)


def relative_entropy_from_densities_with_support(p: tp.Callable,
                                                 q: tp.Callable,
                                                 a: float,
                                                 b: float,
                                                 base: float = np.e,
                                                 eps_abs: float = 1.49e-08,
                                                 eps_rel: float = 1.49e-08
                                                 ) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p
                D_KL(p||q) = E_p [log(p/q)]
    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    def integrand(x: float):
        return _relative_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun)

    return cubature(func=integrand,
                    ndim=1,
                    fdim=1,
                    xmin=np.array([a]),
                    xmax=np.array([b]),
                    vectorized=False,
                    adaptive='p',
                    abserr=eps_abs,
                    relerr=eps_rel)[0].item()


################################################################################
# Relative Entropy (KL Divergence)
################################################################################
def _relative_entropy_integrand(p: tp.Callable,
                                q: tp.Callable,
                                x: float,
                                log_fun: tp.Callable = np.log) -> float:
    """
    Compute the integrand p(x) * log(p(x) / q(x)) at a given point x for the calculation of relative
    entropy.
    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    Returns
    -------
    Integrand for the relative entropy calculation
    """
    qx = q(x)
    px = p(x)
    if qx == 0.0:
        if px == 0.0:
            return 0.0
        else:
            raise ValueError(f'q(x) is zero at x={x} but p(x) is not')
    elif px == 0.0:
        return 0.0
    else:
        return px * log_fun(px / qx)


def continuous_cross_entropy_from_sample(sample_p: np.ndarray,
                                         sample_q: np.ndarray,
                                         base: float = np.e,
                                         eps_abs: float = 1.49e-08,
                                         eps_rel: float = 1.49e-08) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p
                H_q(p) = - E_p [log(q)]
    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return cross_entropy_from_kde(kde_p, kde_q, base=base, eps_abs=eps_abs, eps_rel=eps_rel)


def cross_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                           q: sm.nonparametric.KDEUnivariate,
                           base: float = np.e,
                           eps_abs: float = 1.49e-08,
                           eps_rel: float = 1.49e-08) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p
                H_q(p) = - E_p [log(q)]
    given by the statsmodels kde objects via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    if not _does_support_overlap(p, q):
        raise ValueError('The support of p and q does not overlap.')

    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))

    return cross_entropy_from_densities_with_support(p=p.evaluate,
                                                     q=q.evaluate,
                                                     a=a,
                                                     b=b,
                                                     base=base,
                                                     eps_abs=eps_abs,
                                                     eps_rel=eps_rel)


def cross_entropy_from_densities_with_support(p: tp.Callable,
                                              q: tp.Callable,
                                              a: float,
                                              b: float,
                                              base: float = np.e,
                                              eps_abs: float = 1.49e-08,
                                              eps_rel: float = 1.49e-08) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p
                H_q(p) = - E_p [log(q)]
    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    return - cubature(func=lambda x: _cross_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
                      ndim=1,
                      fdim=1,
                      xmin=np.array([a]),
                      xmax=np.array([b]),
                      vectorized=False,
                      adaptive='p',
                      abserr=eps_abs,
                      relerr=eps_rel)[0].item()


################################################################################
# Cross Entropy
################################################################################
def _cross_entropy_integrand(p: tp.Callable,
                             q: tp.Callable,
                             x: float,
                             log_fun: tp.Callable) -> float:
    """
    Compute the integrand p(x) * log(q(x)) at a given point x for the calculation of cross entropy.
    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    Returns
    -------
    Integrand for the cross entropy calculation
    """
    # return p(x) * log_fun(q(x) + 1e-12)
    qx = q(x)
    px = p(x)
    if qx == 0.0:
        if px == 0.0:
            return 0.0
        else:
            raise ValueError(f'q(x) is zero at x={x} but p(x) is not')
    elif px == 0.0:
        return 0.0
    else:
        return px * log_fun(qx)


def _select_vectorized_log_fun_for_base(base: float, gpu: bool = False) -> tp.Callable:
    if base == 2:
        return np.log2
    if base == np.e:
        return np.log
    if base == 10:
        return np.log10

    raise ValueError('base not supported')


def _does_support_overlap(p: sm.nonparametric.KDEUnivariate,
                          q: sm.nonparametric.KDEUnivariate) -> bool:
    """
    Determine whether the support of distributions of kernel density estimates p and q overlap.
    Parameters
    ----------
    p: statsmodels kde object representing an approximation of the distribution p
    q: statsmodels kde object representing an approximation of the distribution q
    Returns
    -------
    whether the support of distributions of kernel density estimates p and q overlap
    """
    return intersection(min(p.support), max(p.support), min(q.support), max(q.support)) is not None


def intersection(a0: float,
                 b0: float,
                 a1: float,
                 b1: float) \
        -> tp.Optional[tp.Tuple[float, float]]:
    """
    Calculate the intersection of two intervals [a0, b0] and [a1, b1]. If the intervals do not
    overlap the function returns None. The parameters must satisfy a0 <= b0 and a1 <= b1.
    Parameters
    ----------
    a0: beginning of the first interval
    b0: end of the first interval
    a1: beginning of the second interval
    b1: end of the second interval
    Returns
    -------
    """
    assert a0 <= b0
    assert a1 <= b1

    if a0 >= b1:
        return None

    if b0 < a1:
        return None

    return max(a0, a1), min(b0, b1)


################################################################################
# Jensen-Shannon Divergence
###############################################################################
def _relative_entropy_from_densities_with_support_for_shannon_divergence(
        p: tp.Callable,
        q: tp.Callable,
        a: float,
        b: float,
        log_fun: tp.Callable = np.log,
        eps_abs: float = 1.49e-08,
        eps_rel: float = 1.49e-08) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p
                D_KL(p||q) = E_p [log(p/q)]
    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """

    def integrand(x):
        return p(x) * log_fun(p(x) / q(x)) if p(x) > 0.0 else 0.0

    return cubature(func=integrand,
                    ndim=1,
                    fdim=1,
                    xmin=np.array([a]),
                    xmax=np.array([b]),
                    vectorized=False,
                    adaptive='p',
                    abserr=eps_abs,
                    relerr=eps_rel)[0].item()


def jensen_shannon_divergence_from_densities_with_support(p: tp.Callable,
                                                          q: tp.Callable,
                                                          a: float,
                                                          b: float,
                                                          base: float = np.e,
                                                          eps_abs: float = 1.49e-08,
                                                          eps_rel: float = 1.49e-08) \
        -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q
                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)
    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    m = lambda x: 0.5 * (p(x) + q(x))
    D_PM = _relative_entropy_from_densities_with_support_for_shannon_divergence(
        p=p,
        q=m,
        a=a,
        b=b,
        log_fun=log_fun,
        eps_abs=eps_abs,
        eps_rel=eps_rel)

    D_QM = _relative_entropy_from_densities_with_support_for_shannon_divergence(
        p=q,
        q=m,
        a=a,
        b=b,
        log_fun=log_fun,
        eps_abs=eps_abs,
        eps_rel=eps_rel)

    return 0.5 * D_PM + 0.5 * D_QM


def jensen_shannon_divergence_from_kde(p: sm.nonparametric.KDEUnivariate,
                                       q: sm.nonparametric.KDEUnivariate,
                                       base: float = np.e,
                                       eps_abs: float = 1.49e-08,
                                       eps_rel: float = 1.49e-08) \
        -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q
                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)
    given by the statsmodels kde objects via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.
    """
    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))
    return jensen_shannon_divergence_from_densities_with_support(p=p.evaluate,
                                                                 q=q.evaluate,
                                                                 a=a,
                                                                 b=b,
                                                                 base=base,
                                                                 eps_abs=eps_abs,
                                                                 eps_rel=eps_rel)


def continuous_jensen_shannon_divergence_from_sample(sample_p: np.ndarray,
                                                     sample_q: np.ndarray,
                                                     base: float = np.e,
                                                     eps_abs: float = 1.49e-08,
                                                     eps_rel: float = 1.49e-08) -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q
                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)
    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.
    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration
    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.
    """
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return jensen_shannon_divergence_from_kde(kde_p,
                                              kde_q,
                                              base=base,
                                              eps_abs=eps_abs,
                                              eps_rel=eps_rel)
