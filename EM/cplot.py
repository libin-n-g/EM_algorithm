import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import metrics

def main():
    if(len(sys.argv)!=4):
        print("Invalid input number. Example execution: python3 cplot <responsibility file> <data file> <label file>")
        return
    else:
        X = np.genfromtxt(sys.argv[1], delimiter='\t',skip_header=1)
        Y = np.genfromtxt(sys.argv[3], delimiter=' ')
        (n,d) = X.shape
        print(X)
        pred = X.argmax(axis=0)
        ax = plt.gca()
        samples = np.genfromtxt(sys.argv[2], delimiter=' ')
        print(metrics.adjusted_rand_score(pred[:d-1],Y[:,2] ))
        ax.scatter(samples[:,0], samples[:,1], c=pred[:d-1], alpha=0.8)
        plt.show()
    return

if __name__ == '__main__':
	main()