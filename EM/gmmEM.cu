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
#include <cuda.h>
#include "common.h"
#include "matrix.cuh"
//using namespace af;

#define QD 4
#define QM 32
#define QT 8

/* S2 * S3 GRID
 * X => T * D Matrix of input (stored as one dimensional)
 * respon => M * T Matrix which store posterior (responsibility matrix)
 * dim => dimension of Input data
 * n => number of datapoints
 * eps => M * D matrix containing first moments
 * eps_sq => M * D matrix containing second moments
 * S1 =>
 */
__global__  void eps_kernal(double* X, double *respon,
		int dim, int n, double* eps, double* eps_sq, double* c, int S1)
{
	int j_ = blockIdx.y;
	int del_j = j_ * QD;
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
    			o[i][j] = X[(q*QT + i)*dim + del_j + j];
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
 * S1 * S2 GRID
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
		temp=id_x*QT*M+M*i;
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
	int temp;
	for (int i = 0; i < QT; ++i) {
		total=0;
		temp=id_x*QT*M+M*i;
    	for (int j = 0; j < M; ++j) {
    		total=total+gamma_hat[temp+j];
		}
		for (int j = 0; j < M; ++j) {
    		gamma[temp+j]=exp(gamma_hat[temp+j]-total);
		}
	}
}
/*
 * Structure for returning data from file
 */
struct Inputdata {
	double * X;
	int n;
	int d;
};
/*
 * READS FILE FOR INPUT DATA
 * FILE FORMAT (n => number of points , d => dimension of each point)
 * n d
 * x_11 ..... x_1d
 * .
 * .
 * x_n1 ..... x_nd
 */

struct Inputdata read_file (const char* file_name)
{
  FILE* file = fopen (file_name, "r");
  check(file, "File %s could not be opened \n", file_name);
  int n, d;
  fscanf (file, "%d %d", &n, &d);
  double *X;
  X = (double *)calloc(n*d, sizeof(double));
  check(X, "Memory allocation for X(Input Data) failed");
  int i = 0;
  while (!feof (file)){
	  for (int j = 0; j < d; ++j) {
		  fscanf (file, "%d", &X[i*d + j]);
	  }
	  i++;
  }
  check((i >= n), "Error in reading Data \n Please check the data format\n");
  struct Inputdata ret;
  ret.X = X;
  ret.n = n;
  ret.d = d;
  fclose (file);
  return ret;
}
/*
 * Format of argv
 * <input filename>  <number of clusters>
 */
int main(int argc, char **argv) {
	double X[5000][50];
	check(argc > 1, "Please give Input Filename as first argument \n");
	check(argc > 2, "Please give number of clusters as second argument \n");
	int M = atoi(argv[2]);
	//File processing
	struct Inputdata input = read_file(argv[1]);
	int D = input.d;
	int N = input.n;

	double *d_gamma, *d_X, *d_gamma_hat, *d_w, *d_sigma, *d_mu;
	int n_block_x = ceil(N*1.0/QT);
    int n_block_y = ceil(M*1.0/QM);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_gamma, sizeof(double) * N *M));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_X, N*D* sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_gamma_hat, N*M* sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_w, M* sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mu, M*D* sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sigma, M*D* sizeof(double)));

    dim3 dimBlock(n_block_x,n_block_y);
    CUDA_SAFE_CALL(cudaMemcpy(d_X, X, N * D * sizeof(double), cudaMemcpyHostToDevice));
    
    calc_log_gamma<<<dimBlock,QM,sizeof(float)*D*QT>>>(d_X,d_gamma, d_mu, d_sigma, d_w, D, M);
	calc_likelihood<<<n_block_x,QT>>>(d_gamma, d_gamma_hat, M);
	normilize<<<n_block_x,QT>>>(d_gamma,M);
	
	CUDA_SAFE_CALL(cudaFree(d_gamma));
	CUDA_SAFE_CALL(cudaFree(d_X));
	CUDA_SAFE_CALL(cudaFree(d_gamma_hat));
	CUDA_SAFE_CALL(cudaFree(d_w));
	CUDA_SAFE_CALL(cudaFree(d_mu));
	CUDA_SAFE_CALL(cudaFree(d_sigma));


	return 0;
}
