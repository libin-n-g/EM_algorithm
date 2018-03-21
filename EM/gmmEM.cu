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

#define QD 10
#define QM 10
#define QT 10
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
