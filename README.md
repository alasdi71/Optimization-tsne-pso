# improved_tsne_pso.py
# تمثيل أداء الخوارزمية المحسنة (t-SNE مع PSO)
# يتطلب: numpy, matplotlib, scikit-learn, scipy

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)

##########################################
# دالة الجودة Qm(k)
##########################################
def quality_Qm(X_high, Y_low, k=10):
    n = X_high.shape[0]
    nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    idx_high = nbrs_high.kneighbors(return_distance=False)[:, 1:]
    nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(Y_low)
    idx_low = nbrs_low.kneighbors(return_distance=False)[:, 1:]
    preserved = []
    for i in range(n):
        set_h = set(idx_high[i])
        set_l = set(idx_low[i])
        preserved.append(len(set_h & set_l) / float(k))
    return np.mean(preserved)

##########################################
# دالة قياس زمن التنفيذ
##########################################
def cpu_time_function(method_func, *args, **kwargs):
    start = time.time()
    result = method_func(*args, **kwargs)
    elapsed_time = time.time() - start
    return result, elapsed_time

##########################################
# حساب مصفوفة P (تشابه عالي الأبعاد)
##########################################
def Hbeta(Di, beta):
    P = np.exp(-Di * beta)
    P_sum = np.sum(P) + 1e-12
    H = np.log(P_sum) + beta * np.sum(Di * P) / P_sum
    return H, P / P_sum

def compute_P(X, perplexity=30.0, tol=1e-5):
    (n, _) = X.shape
    D = squareform(pdist(X, 'sqeuclidean'))
    P = np.zeros((n, n))
    logU = np.log(perplexity)
    for i in range(n):
        Di = D[i, np.arange(n) != i]
        beta = 1.0
        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        betamin, betamax = -np.inf, np.inf
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta
                beta = (beta*2.0 if betamax in [-np.inf, np.inf] else (beta+betamax)/2.0)
            else:
                betamax = beta
                beta = (beta/2.0 if betamin in [-np.inf, np.inf] else (beta+betamin)/2.0)
            H, thisP = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1
        P[i, np.arange(n) != i] = thisP
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    return P

##########################################
# PSO مبسط لتحسين مواضع النقاط
##########################################
class SimplePSO:
    def __init__(self, P, n_particles=20, dim=2, iters=100, w=0.72, c1=1.49, c2=1.49):
        self.P = P
        self.n = P.shape[0]
        self.dim = dim
        self.n_particles = n_particles
        self.iters = iters
        self.w, self.c1, self.c2 = w, c1, c2

    def _score(self, Y):
        sum_y = np.sum(np.square(Y), axis=1)
        D = -2.0 * np.dot(Y, Y.T)
        D = D + sum_y[:, None] + sum_y[None, :]
        num = 1.0 / (1.0 + D)
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        return np.sum(self.P * np.log(self.P / Q))

    def optimize(self, Y_init=None):
        D = self.n * self.dim
        base = np.random.randn(self.n, self.dim) * 1e-3 if Y_init is None else Y_init.copy()
        particles = np.array([base.flatten() + np.random.randn(D) * 1e-3 for _ in range(self.n_particles)])
        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_scores = np.array([self._score(p.reshape(self.n, self.dim)) for p in particles])
        gbest_idx = np.argmin(pbest_scores)
        gbest, gbest_score = pbest[gbest_idx].copy(), pbest_scores[gbest_idx]
        scores_history = []
        for t in range(self.iters):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(D), np.random.rand(D)
                velocities[i] = (self.w*velocities[i] +
                                 self.c1*r1*(pbest[i]-particles[i]) +
                                 self.c2*r2*(gbest-particles[i]))
                particles[i] += velocities[i]
                Y = particles[i].reshape(self.n, self.dim)
                val = self._score(Y)
                if val < pbest_scores[i]:
                    pbest_scores[i] = val
                    pbest[i] = particles[i].copy()
                    if val < gbest_score:
                        gbest_score, gbest = val, particles[i].copy()
            scores_history.append(gbest_score)
        return gbest.reshape(self.n, self.dim), scores_history

##########################################
# التجربة الرئيسية
##########################################
def demo_compare_tsne_pso(max_samples=500, perplexity=30.0):
    # بيانات Digits
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]

    # تقليل الأبعاد بالـPCA
    Xp = PCA(n_components=30).fit_transform(X)
    P = compute_P(Xp, perplexity=perplexity)

    # t-SNE الأصلي
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', n_iter=500, random_state=0)
    (Y_tsne, tsne_time) = cpu_time_function(tsne.fit_transform, Xp)
    qm_tsne = quality_Qm(Xp, Y_tsne, k=10)

    # PSO-tSNE
    pso = SimplePSO(P, n_particles=15, dim=2, iters=50)
    (pso_output, pso_time) = cpu_time_function(pso.optimize)
    Y_pso, scores_hist = pso_output
    qm_pso = quality_Qm(Xp, Y_pso, k=10)

    # رسومات النتائج
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(Y_tsne[:,0], Y_tsne[:,1], c=y, cmap='tab10', s=10)
    plt.title(f"t-SNE\nTime={tsne_time:.2f}s, Qm={qm_tsne:.3f}")
    plt.subplot(1,2,2)
    plt.scatter(Y_pso[:,0], Y_pso[:,1], c=y, cmap='tab10', s=10)
    plt.title(f"PSO-tSNE\nTime={pso_time:.2f}s, Qm={qm_pso:.3f}")
    plt.tight_layout()
    plt.show()

    # منحنى KL divergence
    plt.plot(scores_hist)
    plt.xlabel("Iteration")
    plt.ylabel("KL divergence")
    plt.title("PSO KL divergence history")
    plt.show()

    # منحنى Qm(k)
    ks = [1,5,10,20,30,50]
    qm_tsne_list = [quality_Qm(Xp, Y_tsne, k=k) for k in ks]
    qm_pso_list  = [quality_Qm(Xp, Y_pso, k=k) for k in ks]
    plt.plot(ks, qm_tsne_list, '-o', label='t-SNE')
    plt.plot(ks, qm_pso_list, '-o', label='PSO-tSNE')
    plt.xlabel("k (nearest neighbors)")
    plt.ylabel("Qm(k)")
    plt.title("Quality measure comparison")
    plt.legend()
    plt.show()

    # مقارنة أزمنة التنفيذ
    plt.bar(["t-SNE", "PSO-tSNE"], [tsne_time, pso_time])
    plt.ylabel("Time (s)")
    plt.title("CPU time comparison")
    plt.show()

    return {
        "tsne_time": tsne_time, "qm_tsne": qm_tsne,
        "pso_time": pso_time, "qm_pso": qm_pso,
        "scores_history": scores_hist
    }

##########################################
# Run
##########################################
if __name__ == "__main__":
    results = demo_compare_tsne_pso(max_samples=500, perplexity=30)
    print("Results:", results)
