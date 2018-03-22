from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

def fit_samples(samples,n_comp):
    start_time = time.time()
    gmix = mixture.GMM(n_components=n_comp, covariance_type='full')
    gmix.fit(samples)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(gmix.means_)
    colors = gmix.predict(samples)
    ax = plt.gca()
    ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
    plt.show()

X= np.genfromtxt(sys.argv[1], delimiter=',')
fit_samples(X,int(sys.argv[2]))