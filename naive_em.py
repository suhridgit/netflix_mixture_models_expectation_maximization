"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    def gaussian(x, mean, var):
        d = len(x)
        logP = -d / 2.0 * np.log(2 * np.pi * var) - 0.5 * np.sum((x - mean) ** 2) / var
        return np.exp(logP)

    n,d = np.shape(X)
    mu,var,P = mixture
    K = len(P)
    post = np.zeros((n, K)) # posterior probabilities to compute
    LL = 0.0  # the LogLikelihood
    for n in range(len(X)):
        for m in range(K):
            post[n,m] = P[m]*gaussian(X[n,:],mu[m,:],var[m])

        total = sum(post[n, :])
        post[n,:] = post[n,:]/total
        LL += np.log(total)

    return (post, LL)

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d = X.shape
    _,k = post.shape
    n_clusters = np.einsum("ij->j",post)
    p_weights = n_clusters/n
    mu = post.T @ X / n_clusters.reshape(k, 1)
    var = np.zeros(k)
    for i in range(k):
        var[i] = np.sum(post[:,i].T @ (X - mu[i])**2 / (n_clusters[i] * d))
    return GaussianMixture(mu, var, p_weights)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood = None
    while 1:
        post, new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
        if old_log_likelihood is not None:
            if (new_log_likelihood - old_log_likelihood) < 1e-6 * abs(new_log_likelihood):
                break
        old_log_likelihood = new_log_likelihood
    return mixture, post, new_log_likelihood
