from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from sklearn import metrics

def fit_samples(samples,n_comp,Y):
    start_time = time.time()
    gmix = mixture.GaussianMixture(n_components=n_comp, covariance_type='diag')
    gmix.fit(samples)
    elapsed_time = time.time() - start_time
    print("time",elapsed_time)
    print("means",gmix.means_)
    colors = gmix.predict(samples)
    ax = plt.gca()
    ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
    print(metrics.adjusted_rand_score(colors,Y[:,2] ))
    plt.show()

X = np.genfromtxt(sys.argv[1], delimiter=' ')
Y = np.genfromtxt(sys.argv[3], delimiter=' ')
fit_samples(X,int(sys.argv[2]),Y)