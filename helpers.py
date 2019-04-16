from numba.decorators import jit
import numpy as np
    
@jit
def distances(X, v, alpha, N, P, K):
    dists = np.zeros((N, P))
    for i in range(N):
        for p in range(P):
            for k in range(K):    
                dists[i, k] += (X[i, p] - v[k, p]) * (X[i, p] - v[k, p]) * alpha[p]
    return dists

@jit
def M_nk(dists, N, K):
    M_nk = np.zeros((N, K))
    exp = np.zeros((N, dists.shape[1]))
    denom = np.zeros(N)
    for i in range(N):
        for p in range(dists.shape[1]):
            exp[i, p] = np.exp(-1 * dists[i, p])
            denom[i] += exp[i, p]
        for k in range(K):
            if denom[i]:
                M_nk[i, k] = exp[i, k] / denom[i]
            else:
                M_nk[i, k] = exp[i, k] / 1e-6
    return M_nk

@jit    
def M_k(M_nk, N, k):
    M_k = np.zeros(k)
    for j in range(k):
        for i in range(N):
            M_k[j] += M_nk[i, j]
        M_k[j] /= N
    return M_k

@jit        
def x_n_hat(X, M_nk, v, N, P, K):
    x_n_hat = np.zeros((N, P))
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for k in range(K):
                x_n_hat[i, p] += M_nk[i, k] * v[k, p]
            L_x += (X[i, p] - x_n_hat[i, p]) * (X[i, p] - x_n_hat[i, p])
    return x_n_hat, L_x

@jit
def yhat(M_nk, y, w, N, K):
    yhat = np.zeros(N)
    L_y = 0.0
    for i in range(N):
        for k in range(K):
            yhat[i] += M_nk[i, k] * w[k]
        yhat[i] = 1e-6 if yhat[i] <= 0 else yhat[i]
        yhat[i] = 0.999 if yhat[i] >= 1 else yhat[i]
        L_y += -1 * y[i] * np.log(yhat[i]) - (1.0 - y[i]) * np.log(1.0 - yhat[i])
    return yhat, L_y

@jit
def LFR(params, data_sensitive, data_nonsensitive, y_sensitive, 
        y_nonsensitive,  k=10, A_x = 1e-4, A_y = 0.1, A_z = 1000, results=0):
    
    LFR.iters += 1 
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape
    
    alpha0 = params[:P]
    alpha1 = params[P : 2 * P]
    w = params[2 * P : (2 * P) + k]
    v = np.matrix(params[(2 * P) + k:]).reshape((k, P))
        
    dists_sensitive = distances(data_sensitive, v, alpha1, Ns, P, k)
    dists_nonsensitive = distances(data_nonsensitive, v, alpha0, Nns, P, k)

    M_nk_sensitive = M_nk(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dists_nonsensitive, Nns, k)
    
    M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
    M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)
    
    L_z = 0.0
    for j in range(k):
        L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])

    x_n_hat_sensitive, L_x1 = x_n_hat(data_sensitive, M_nk_sensitive, v, Ns, P, k)
    x_n_hat_nonsensitive, L_x2 = x_n_hat(data_nonsensitive, M_nk_nonsensitive, v, Nns, P, k)
    L_x = L_x1 + L_x2

    yhat_sensitive, L_y1 = yhat(M_nk_sensitive, y_sensitive, w, Ns, k)
    yhat_nonsensitive, L_y2 = yhat(M_nk_nonsensitive, y_nonsensitive, w, Nns, k)
    L_y = L_y1 + L_y2

    criterion = A_x * L_x + A_y * L_y + A_z * L_z

    if LFR.iters % 250 == 0:
        print(LFR.iters, criterion)
      
    if results:
        return yhat_sensitive, yhat_nonsensitive, M_nk_sensitive, M_nk_nonsensitive
    else:
        return criterion
LFR.iters = 0
