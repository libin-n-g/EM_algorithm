import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    if(len(sys.argv)!=3):
        print("Invalid input number. Example execution: python3 cplot <responsibility file> <data file>")
        return
    else:
        X = np.genfromtxt(sys.argv[1], delimiter='\t',skip_header=1)
        pred = X.argmax(axis=0)
        ax = plt.gca()
        samples = np.genfromtxt(sys.argv[2], delimiter=',')
        ax.scatter(samples[:,0], samples[:,1], c=pred, alpha=0.8)
        plt.show()
    return

if __name__ == '__main__':
	main()