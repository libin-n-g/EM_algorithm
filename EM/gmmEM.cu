/*
 * gmmEm.cu
 *
 *  Created on: 17-Mar-2018
 *      Author: libin
 */

//#include <arrayfire.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "matrix.cuh"
//using namespace af;

#define QD 4
#define QM 32
#define QT 8
#define S1 10
#define S2 10
#define S3 10
/*
 * M*D number of blocks
 */
__global__  void eps_kernal(double* O, double *respon,
		int dim, int n, double* eps, double* eps_sq, double* c)
{
	int j_ = blockIdx.y;
	//int i_ = blockIdx.x;
	int del_j = j_ * QD;
	//int row =  threadIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    double c_m = 0;
    double eps_[QD];
    double eps_sq_[QD];
    __shared__ double o[QT][QD];
    for (int i = 0; i < QD; ++i) {
    	eps_[i] = 0;
    	eps_sq[i] = 0;
	}
    for (int q = 0; q < S1; ++q) {
    	for (int i = 0; i < QT; ++i) {
    		for (int j = 0; j < QD; ++j) {
    			o[i][j] = O[(q*QT + i)*dim + del_j + j];
			}
		}
//#pragma unroll
		for (int t = 0; t < QT; ++t) {
			c_m = c_m + respon[q*QT + t];
			for (int d = 0; d < QD; ++d) {
				eps_[d] = eps_[d] + (respon[q*QT + t]* o[t][d]);
				eps_sq_[d] = eps_sq_[d] + (respon[q*QT + t]* pow(o[t][d], 2));
			}
		}
	}
    for (int i = 0; i < QD; ++i) {
    	eps[m*QM + i] = eps_[i];
    	eps_sq[ m *QM + i] = eps_sq_[i];
    }
    c[m] = c_m;
    __syncthreads();
}

__device__ double square(double x){
    return x*x;
}
/*
 * d -> dimension of data(feature) == D
 * X -> T * D Matrix of input (stored as one dimensional)
 * gamma -> M * N array (N -> number of data points) (responsibility )
 * mu -> M -dimensional array (M components)
 * sigma -> M * D dimensional array ( each row represents diagonal coefficients for each sigma)
 * w -> M dimensional array (mixing coefficient)
 * M -> number of components
 */

__global__ void calc_gamma(double * X, double * gamma, double* mu,
		double* sigma, double* w, int d, int M){
    __shared__ double x[QT][100]; //assuming d < 100
    int id_x = blockIdx.x;
    int id_y = blockIdx.y;
    // Coping to Shared Memory
    for (int i = 0; i < QT; ++i) {
    	for (int j = 0; j < d; ++j) {
    		x[i][j] = X[(id_x + i)* QT  + j];
		}
	}

    int i,j;
    double total;

    int temp = id_x * QT * d;

    int m = blockIdx.y * QM + threadIdx.y;
    double gamma_[QT];
    // NOTE : assuming diagonal variance matrix
    double det = 1;
    for (int i = 0; i < d; ++i) {
		det = det * sigma[ d*m + i];
	}
    double Gm = log(w[m]) + 0.39909*d + 0.5*log(det);
//    for(i=0;i<Qt;i++){
//        for(j=0;j<d;j++){
//            x[i*d+j]=X[temp+i*d+j];
//        }
//    }
    __syncthreads();

    for(i=0;i<QT;i++){
        gamma_[i] = Gm;
    }

    for(i=0;i<QT;i++){
        total=0;
        for(j=0;j< d;j++){
            total = total+square(x[i][j]-mu[m*d+j])/square(sigma[m*d+j]);
        }
        gamma_[i] = gamma_[i]+total;
    }
    for (int k = 0; k < QT; ++k) {
    	gamma[m*QM + k] = gamma_[k];
	}
}
int main(int argc, char **argv) {
	float *M = NULL;
	float x = 0;
	M = (float *)malloc(9 * sizeof(float));
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (i==j)
				M[i*3  + j]=1;
			else
				M[i*3  + j]=j;
		}
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			printf("%f\t", M[i*3  + j]);
		}
		printf("\n");
	}

	//af_det(&x,&y,M);
	x = determinant(M,3);
	printf("DET = %f", x);
	return 0;
}
