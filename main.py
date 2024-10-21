import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here

Ks = [1,2,3,4]

seeds = [0,1,2,3,4]

# =============================================================================
# 2. K-means
# =============================================================================

for k in Ks:
    for seed in seeds:
        mixture, post = common.init(X, k, seed = seed) # Initialize K-means
        mixture, post, cost = kmeans.run(X,mixture,post)
        common.plot(X,mixture,post,[k,seed])
        print(k,seed,cost)

# =============================================================================
# 3. Expectation–maximization algorithm
# =============================================================================

# E_step
mixture, post = common.init(X, 3, seed=0)
mu, var, p = mixture
post, log_likelihood = naive_em.estep(X, mixture)

# M_step

mu, var, p_weights = naive_em.mstep(X,post)

# Run

mixture, post = common.init(X, 3, seed=0)
mixture,post,LL = naive_em.run(X, mixture,post)

# =============================================================================
# 4. Comparing K-means and EM
# =============================================================================

for K in Ks:
    for seed in seeds:
        mixture, post = common.init(X, K = K, seed = seed) # Initialize K-means
        mixture, post, log_likelihood = naive_em.run(X, mixture, post)
        common.plot(X, mixture, post, [K, seed])
        print(K, seed, log_likelihood)

# =============================================================================
# 5. Bayesian Information Criterion
# Picking the best K
# =============================================================================

for K in Ks:
    mixture, post = common.init(X, K = K) # Initialize K-means
    mixture, post, log_likelihood = naive_em.run(X, mixture, post)
    BIC = common.bic(X, mixture, log_likelihood)
    print(K, BIC)


# =============================================================================
# 7. Implementing EM for matrix completion
# Test for comlete case
# =============================================================================

X = np.loadtxt("toy_data.txt")
Ks = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
mixture, post = common.init(X, 3, seed=0)

post2, log_likelihood2 = em.estep(X, mixture)
Mu2,Var2,P2 = em.mstep(X,post2,mixture)