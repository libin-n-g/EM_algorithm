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

/* S2 * S3 GRID QM threads per block
 * X => T * D Matrix of input (stored as one dimensional)
 * respon => M * T Matrix which store posterior (responsibility matrix)
 * dim => dimension of Input data
 * n => number of data points
 * eps => M * D matrix containing first moments
 * eps_sq => M * D matrix containing second moments
 * S1 =>
 * c => sum of responsibilities for each components
 */
__global__ void eps_kernal(double* X, double *respon, int dim, int n,
		double* eps, double* eps_sq, double* c, int S1) {
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
				o[i][j] = X[(q * QT + i) * dim + del_j + j];
			}
		}
//#pragma unroll
		for (int t = 0; t < QT; ++t) {
			c_m = c_m + respon[q * QT + t];
			for (int d = 0; d < QD; ++d) {
				eps_[d] = eps_[d] + (respon[q * QT + t] * o[t][d]);
				eps_sq_[d] = eps_sq_[d]
						+ (respon[q * QT + t] * pow(o[t][d], 2));
			}
		}
	}
	for (int i = 0; i < QD; ++i) {
		eps[m * QM + i] = eps_[i];
		eps_sq[m * QM + i] = eps_sq_[i];
	}
	c[m] = c_m;
	__syncthreads();
}

__device__ double square(double x) {
	return x * x;
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
		double* sigma, double* w, int d, int M) {
	extern __shared__ double x[];
	/* memory size should be give as 3rd parameter
	 when calling the kernel */
	int id_x = blockIdx.x;
	// Coping to Shared Memory
	for (int i = 0; i < QT; ++i) {
		for (int j = 0; j < d; ++j) {
			x[i * d + j] = X[(id_x + i) * QT + j];
		}
	}

	int i, j;
	double total;

	int m = blockIdx.y * QM + threadIdx.x; //
	double gamma_[QT];
	/*
	 * WARNING : assuming diagonal variance matrix
	 */
	double det = 1;
	for (int i = 0; i < d; ++i) {
		det = det * sigma[d * m + i];
	}
	// log(2*pi)*0.5 = 0.399089934
	double Gm = log(w[m]) + 0.399089934 * d + 0.5 * log(det);
	__syncthreads();

	for (i = 0; i < QT; i++) {
		gamma_[i] = Gm;
	}

	for (i = 0; i < QT; i++) {
		total = 0;
		for (j = 0; j < d; j++) {
			total = total
					+ square(x[i * d + j] - mu[m * d + j])
							/ square(sigma[m * d + j]);
		}
		gamma_[i] = gamma_[i] + total;
	}
	for (int k = 0; k < QT; ++k) {
		gamma_hat[m * QM + k] = gamma_[k];
	}
}

/* 
 * normalize gamma
 * number of blocks => S1*QT
 * number of threads per block => M
 * gamma => unnormalized responsibilities M * T array
 */
__global__ void normilize(double * gamma, double *log_like, int M, int N) {
	int id_x = blockIdx.x * gridDim.y + blockIdx.y;
	int j = threadIdx.x;
	gamma[j * M + id_x] = exp(gamma[id_x + j * M] - log_like[j]);
}
/*
 * QT*S1 => Number of blocks
 * number of threads => M
 * M => number of components
 * N => number of points
 * gamma => N length array of log likelihoods
 * gamma_hat => M * N matrix of responsibility
 */
__global__ void calc_likelihood(double * gamma, double * gamma_hat, int M,
		int N) {
	int id_x = blockIdx.y * gridDim.x + blockIdx.x;
	__shared__ double total;
	// coping to shared memory
	if (threadIdx.x == 0) {
		total = 0;
	}
	__syncthreads();
	if (id_x < N) {
		total = total + exp(gamma_hat[M * threadIdx.x + id_x]);
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		gamma[id_x] = log(total);
	}
//	for (int i = 0; i < QT; ++i) {
//		total=0;
//		//temp=id_x*QT*M+M*i;
//    	for (int j = 0; j < M; ++j) {
//    		total += (gamma_hat[temp+j]);
//		}
//		for (int j = 0; j < M; ++j) {
//    		gamma[temp+j] = exp(gamma[temp+j]-(total));
//		}
//	}
}
/*
 * eps M * D
 * eps_sq M * D
 * T number of points
 * M number of components
 * D feature dimension
 * mu M * D (OUT)
 * sigma M * D (OUT)
 * w -> mixing coefficient M dimensional (OUT)
 */

__global__ void find_mu_sigma_omega(double* eps,double* eps_sq, double* gamma, int T,int M, int D,
		double * mu, double *sigma, double *w){
	int component = threadIdx.x;
	double cm = 0;
	for (int i = 0; i < T; ++i) {
		cm = cm + gamma[T*component + i];
	}
	w[component] = cm / ((double)T);
	for (int i = 0; i < D; ++i) {
		mu[component*D + i] = eps[component*D + i] / cm;
	}
	for (int i = 0; i < D; ++i) {
		sigma[D*component + i] = eps_sq[component*D + i]/ cm  - mu[component*D + i]*mu[component*D + i];
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
 * x_11 ..... x_1d
 * .
 * .
 * x_n1 ..... x_nd
 */

struct Inputdata read_file(const char* file_name, int n, int d) {
	FILE* file = fopen(file_name, "r");
	check(file, "File %s could not be opened \n", file_name);
	double *X;
	X = (double *) calloc(n * d, sizeof(double));
	check(X, "Memory allocation for X(Input Data) failed");
	int i = 0;
	while (!feof(file)) {
		for (int j = 0; j < d; ++j) {
			fscanf(file, "%lf", &X[i * d + j]);
		}

		i++;
	}
	check((i >= n), "Error in reading Data \n Please check the data format\n");
	struct Inputdata ret;
	ret.X = X;
	ret.n = n;
	ret.d = d;
	fclose(file);
	return ret;
}
/*
 * Format of argv
 * <input filename>  <number of clusters>
 */
int main(int argc, char **argv) {
	double X[5000][50];
	check(argc > 1, "Please give Input Filename as first argument \n");
	check(argc > 2, "Please give number of points as second argument \n");
	check(argc > 3,
			"Please give dimensions of point (number of features) as third argument \n");
	check(argc > 4,
			"Please give number of clusters (components) as forth argument \n");
	int N = atoi(argv[2]);
	int D = atoi(argv[3]);
	int M = atoi(argv[4]);
	//File processing
	struct Inputdata input = read_file(argv[1], N, D);

	double *d_loglike, *d_X, *d_gamma, *d_w, *d_sigma, *d_mu, *d_c, *d_eps,
			*d_eps_sq;
	int S1 = ceil(N * 1.0 / QT);
	int S2 = ceil(M * 1.0 / QM);
	int S3 = ceil(D * 1.0 / QD);
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_c, sizeof(double) * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_eps, sizeof(double) * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_eps_sq, sizeof(double) * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_loglike, sizeof(double) * N * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_X, N * D * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_gamma, N * M * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_w, M * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_mu, M * D * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_sigma, M * D * sizeof(double)));

	dim3 dimBlock(S1, S2);
	dim3 dimBlock2(S2, S3);
	dim3 normalize_Block(S1, QT);
	CUDA_SAFE_CALL(
			cudaMemcpy(d_X, X, N * D * sizeof(double), cudaMemcpyHostToDevice));
	/*
	 * TODO : Make loop checking log likelihood
	 * (see Fast Estimation of Gaussian Mixture Model Parameters on GPU using CUDA)
	 * TODO : get log likelihood from d_loglike
	 */

	calc_log_gamma<<<dimBlock, QM, sizeof(float) * D * QT>>>(d_X, d_loglike,
			d_mu, d_sigma, d_w, D, M);
	calc_likelihood<<<S1, QT>>>(d_loglike, d_gamma, M, N);
	normilize<<<normalize_Block, M>>>(d_gamma, d_loglike, M, N);
	eps_kernal<<<dimBlock2, QM>>>(d_X, d_gamma, D, N, d_eps, d_eps_sq, d_c, S1);
	find_mu_sigma_omega<<< 1, M>>>(d_eps, d_eps_sq, d_gamma, N, M, D, d_mu, d_sigma, d_w);
	CUDA_SAFE_CALL(cudaFree(d_loglike));
	CUDA_SAFE_CALL(cudaFree(d_X));
	CUDA_SAFE_CALL(cudaFree(d_gamma));
	CUDA_SAFE_CALL(cudaFree(d_w));
	CUDA_SAFE_CALL(cudaFree(d_mu));
	CUDA_SAFE_CALL(cudaFree(d_sigma));

	return 0;
}
