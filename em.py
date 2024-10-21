"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    def log_gaussian(x, mean, var):
        d = len(x)
        logP = -(d / 2.0) * np.log(2 * np.pi * var) - 0.5 * np.sum((x - mean) ** 2) / var
        return logP

    n,d = X.shape
    Mu, Var, P = mixture
    K = len(P)
    post = np.zeros((n,K))
    LL = 0.0

    for t in range(n):
        mask = (X[t,:] != 0)
        for j in range(K):
            post[t,j] = np.log(P[j] + 1.0e-16) + log_gaussian(X[t,mask],Mu[j,mask],Var[j])
        total = logsumexp(post[t,:])
        LL += total
        post[t,:] = post[t,:] - total

    if K == 1:
        post = np.ones([n,1])

    return (np.exp(post),LL)





def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d = X.shape
    Mu,Var, P = mixture
    K = len(P)
    num_points = np.sum(post,axis=0)
    min_variance = 0.25
    P = (num_points/np.sum(num_points))
    for j in range(K):
        sse , weight = 0.0, 0.0
        for i in range(d):
            mask = (X[:,i] != 0)
            n_sum = np.sum(post[mask,j])

            if (n_sum >=1.0):
                Mu[j,i] = np.dot(post[mask,j],X[mask,i])/n_sum
            sse += np.dot(post[mask,j],(X[mask,i]-Mu[j,i])**2)
            weight += n_sum

        Var[j] = sse/weight
    for j in range(K):
            Var[j] = max([min_variance,Var[j]])

    return GaussianMixture(Mu, Var, P)


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
    prevLL = None
    while True:
        post, newLL = estep(X,mixture)
        mixture = mstep(X,post,mixture)

        if prevLL is not None:
            if (newLL - prevLL) < 1e-6 * abs(newLL): break
        prevLL = newLL
    return mixture,post,newLL



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    def log_gaussian(x, mean, var):
        d = len(x)
        logP = -(d / 2.0) * np.log(2 * np.pi * var) - 0.5 * np.sum((x - mean) ** 2) / var
        return logP

    n,d = X.shape
    Xnew = np.copy(X)
    Mu, Var, P = mixture
    K = len(P)
    for u in range(n):
        mask = X[u,:] != 0
        mask0 = X[u,:] == 0
        post = np.zeros(K)

        for j in range(K):
            post[j] = np.log(P[j]) + log_gaussian(X[u,mask],Mu[j,mask],Var[j])
        post = np.exp(post -logsumexp(post))
        Xnew[u,mask0] = np.dot(post,Mu[:,mask0])

    return Xnew



