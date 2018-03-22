/*
 * gmmEm.cu
 *
 *  Created on: 17-Mar-2018
 *      Author: libin,axel
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
 * gamma_hat -> M * N array (N -> number of data points) 
 * mu -> M * D -dimensional array (M components)
 * sigma -> M * D dimensional array ( each row represents diagonal coefficients for each sigma)
 * w -> M dimensional array (mixing coefficient)
 * M -> number of components
 */

__global__ void calc_log_gamma(double * X, double * gamma_hat, double* mu,
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
    	gamma_hat[m*QM + k] = gamma_[k];
	}
}

/* 
* normalize gamma
* total threads = S1 i.e number of classes to which X is divided
* each thread normalizes QT feature vector gammas   
*/
__global__ void normilize(double * gamma, int M){
	int id_x = blockIdx.x;
	double total;
	int temp;
	for (int i = 0; i < QT; ++i) {
		total=0;
		temp=id_X*QT*M+M*i;
    	for (int j = 0; j < M; ++j) {
    		total=total+gamma[temp+j];
		}
		for (int j = 0; j < M; ++j) {
    		gamma[temp+j]=gamma[temp+j]/temp;
		}
	}
}

__global__ void calc_likelihood(double * gamma, double * gamma_hat,int M){
	int id_x = blockIdx.x;
	double total;
	for (int i = 0; i < QT; ++i) {
		total=0;
		temp=id_X*QT*M+M*i;
    	for (int j = 0; j < M; ++j) {
    		total=total+gamma_hat[temp+j];
		}
		for (int j = 0; j < M; ++j) {
    		gamma[temp+j]=exp(gamma_hat[temp+j]-total);
		}
	}
}

int main(int argc, char **argv) {
	FILE *fp;
	int M;
	double X[5000][50];
	//float *M = NULL;
	float x = 0;

	//File processing
	fp = fopen(argv[1], "r");
	if (fp == NULL)
    {
        printf("Could not open file %s", filename);
        return 0;
	}
	int i=0;
	int j=0;
	while(1){
		fscanf(fp, "%lf", X[i][j]);
		c = getc(fp);
		if(c==EOF){
			break;
		} else if(c=='\n'){
			i++;
			j=0;
		} else if(c==','){
			j++;
		}
	}
	int n=i+1;
	int d=j+1;

	// M = (float *)malloc(9 * sizeof(float));
	// for (int i = 0; i < 3; ++i) {
	// 	for (int j = 0; j < 3; ++j) {
	// 		if (i==j)
	// 			M[i*3  + j]=1;
	// 		else
	// 			M[i*3  + j]=j;
	// 	}
	// }
	// for (int i = 0; i < 3; ++i) {
	// 	for (int j = 0; j < 3; ++j) {
	// 		printf("%f\t", M[i*3  + j]);
	// 	}
	// 	printf("\n");
	// }

	// //af_det(&x,&y,M);
	// x = determinant(M,3);
	// printf("DET = %f", x);
	//--------------------------------------------

	double *d_gamma, *d_X, *d_gamma_hat, *d_w, *d_sigma, *d_mu;
	int n_block_x = ceil(n*1.0/QT);
    int n_block_y = ceil(M*1.0/QM);
	cudaMalloc((void **)&d_gamma, n*m* sizeof(double));
	cudaMalloc((void **)&d_X, n*d* sizeof(double));
	cudaMalloc((void **)&d_gamma_hat, n*m* sizeof(double));
	cudaMalloc((void **)&d_w, m* sizeof(double));
	cudaMalloc((void **)&d_mu, m*d* sizeof(double));
	cudaMalloc((void **)&d_sigma, m*d* sizeof(double));

    dim3 dimBlock(n_block_x,n_block_y);
    cudaMemcpy(d_X, X, n * D * sizeof(double), cudaMemcpyHostToDevice);
    calc_log_gamma<<<dimBlock,QM,sizeof(float)*d*Qt>>>(d_X,d_gamma_hat,d_mu,d_sigma,d,M);
	calc_likelihood<<<n_block_x,QT>>>(d_gamma, d_gamma_hat, M);
	normilize<<<n_block_x,QT>>>(d_gamma,M)

	
	cudaFree(d_gamma);
	cudaFree(d_X);
	cudaFree(d_gamma_hat);
	cudaFree(d_w);
	cudaFree(d_mu);
	cudaFree(d_sigma);


	return 0;
}
