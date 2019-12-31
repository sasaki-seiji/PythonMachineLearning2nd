import numpy as np
from sklearn.datasets import make_moons

from rbf_kernel_pca_pair import rbf_kernel_pca

X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)




x_new = X[25]
x_new
# 2019.12.31 add
print('x_new: ', x_new)



x_proj = alphas[25] # original projection
x_proj
# 2019.12.31 add
print('x_proj: ', x_proj)



def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# projection of the "new" datapoint
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj 
# 2019.12.31 add
print('x_reproj: ', x_reproj)