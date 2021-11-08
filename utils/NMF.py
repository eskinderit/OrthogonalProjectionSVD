import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import nnls

def NMF(X, m, T=1000, tau=1e-2, return_error=False):
    n, p = X.shape
    
    # ANLS Algorithm
    W = np.random.uniform(0, 1, (n, m))
    H = np.zeros((m, p))
    t = 0
    
    # Stopping Criterion
    cont = True
    
    # Metrics
    res = np.zeros(T)
    
    while cont:
        
        # Step (a): Update H
        H_0 = H.copy()
        for j in range(p):
            H[:, j], _ = nnls(W, X[:, j])
        
        # Step (b): Update W
        W_0 = W.copy()
        for j in range(n):
            W[j, :], _ = nnls(H.T, X[j, :].T)
        
        # Step (c): Normalize H
        for k in range(m):
            H[k, :] = H[k, :] / np.max(H[k, :])
            
        # Update residual
        res[t] = np.linalg.norm(X - W@H, 'fro')
            
        # Step (d): Check Convergence
        err = np.linalg.norm(W - W_0, 'fro') / (np.linalg.norm(W_0)+1e-7) + np.linalg.norm(H - H_0, 'fro') / (np.linalg.norm(H_0)+1e-7)
        
        if err < tau or t >= T-1:
            if err < tau:
                print("Correct!")
            cont = False
        t = t+1
    res = res[:t]
    
    if return_error:
        return (W, H), res
    return W, H



















