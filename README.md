Mixture Models & Expectation-Maximization Algorithm

## Overview
This repository implements Gaussian Mixture Models (GMM) and Expectation-Maximization (EM) algorithms for various tasks such as matrix completion and collaborative filtering. The code is structured to initialize a mixture model, perform E-step and M-step iterations, and compare K-means clustering with the EM approach.

## Files
common.py: Contains utility functions and data structures for handling Gaussian Mixture models, initializing the models, and visualizing results.

Key Functions:

init(X, K, seed=0): Initializes the Gaussian Mixture Model (GMM) with random points.
plot(X, mixture, post, title): Visualizes the mixture model for 2D data.

em.py: Contains the implementation of the Expectation-Maximization (EM) algorithm.

Key Functions:

estep(X, mixture): Performs the Expectation (E) step of the EM algorithm.
mstep(X, post): Performs the Maximization (M) step of the EM algorithm.
run(X, mixture, post): Executes the EM algorithm for matrix completion.


EM2.py: This file extends the basic EM algorithm from em.py, with additional refinements or specific use cases.

Key Functions:

estep: Similar to the E-step in em.py
mstep: Extended version of the M-step function.


main.py: Contains the main script for running experiments with Gaussian Mixtures, Expectation-Maximization, and K-Means clustering. It compares the performance of K-means and EM, and applies the algorithms to matrix completion problems.

## Key Highlights:

Initializes different models with various parameters such as K (number of clusters) and random seeds.
Visualizes and prints the results, including log-likelihood and Bayesian Information Criterion (BIC).

## Configuration:

Modify the number of clusters (Ks) and seeds for the Gaussian Mixture Models in main.py.
Add your dataset in place of toy_data.txt for real-world experiments.

## Visualization
The code includes built-in functions for plotting the results of the Gaussian Mixture and EM algorithms using Matplotlib. These plots provide insights into how the mixture model is fitting the data.

## Use Cases
Collaborative Filtering: Using GMM and EM for recommendation systems.
Matrix Completion: Filling in missing values in data matrices, commonly used in recommender systems or collaborative filtering.
Clustering: Comparing K-Means clustering with Gaussian Mixtures.
