import typing as tp

import numpy as np
import statsmodels.api as sm
from cubature import cubature


################################################################################
# Jensen-Shannon Divergence
###############################################################################
def jensen_shannon_divergence_from_samples(sample_p: np.ndarray,
                                           sample_q: np.ndarray,
                                           base: float = np.e) -> float:
    return continuous_jensen_shannon_divergence_from_sample(sample_p=sample_p,
                                                            sample_q=sample_q,
                                                            base=base)


def jensen_shannon_divergence_from_densities_with_support(p: tp.Callable,
                                                          q: tp.Callable,
                                                          a: float,
                                                          b: float,
                                                          base: float = np.e,
                                                          eps_abs: float = 1.49e-08,
                                                          eps_rel: float = 1.49e-08) -> float:
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
                                       eps_rel: float = 1.49e-08) -> float:
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


def _select_vectorized_log_fun_for_base(base: float):
    if base == 2:
        return np.log2
    if base == np.e:
        return np.log
    if base == 10:
        return np.log10

    raise ValueError('base not supported')
