from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import sys
def main():
    if(len(sys.argv)!=4):
        print("Invalid input number. Example execute: python3 makeBlobDataset <n_samples> <n_features> <n_centres>")
        return
    else:
        X, y = make_blobs(n_samples=int(sys.argv[1]), centers=int(sys.argv[3]), n_features=int(sys.argv[2]),random_state=0)
        filename="data_"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]
        data = np.hstack((X,np.asmatrix(y).transpose()))
        np.savetxt(filename, data, delimiter=",")
    return

main()
