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
#include <time.h>
//using namespace af;

#define QD 2
#define QM 3
#define QT 100

__device__ double square(double x) {
	return x * x;
}
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
		double* eps, double* eps_sq, double* c, int M, int S1) {
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
		if (threadIdx.x == 0){
			for (int i = 0; i < QT; ++i) {
				for (int j = 0; j < QD; ++j) {
					o[i][j] = X[(q * QT + i) * dim + del_j + j];
				}
			}
		}
		__syncthreads();
//#pragma unroll
		for (int t = 0; t < QT; ++t) {
			c_m = c_m + respon[m *  + t];
			for (int d = 0; d < QD; ++d) {
				eps_[d] = eps_[d] + (respon[ m * dim + t] * o[t][d]);
				eps_sq_[d] = eps_sq_[d]
						+ (respon[ m * dim + t] * square((o[t][d])));
			}
		}
	}
	for (int i = 0; i < QD; ++i) {
		eps[m * dim + i] = eps_[i];
		eps_sq[m * dim + i] = eps_sq_[i];
//		printf("eps %lf %lf %d %d \n", eps_[i], eps_sq_[i], m, i );
	}
//	printf("pi %lf %d \n", c_m, m);
	c[m] = c_m;
	__syncthreads();
}

/*
 * S1 * S2 GRID QM threads per block
 * d -> dimension of data(feature) == D
 * X -> T * D Matrix of input (stored as one dimensional)
 * gamma -> M * N array (N -> number of data points) (responsibility )
 * gamma_hat -> M * N array (N -> number of data points) 
 * mu -> M * D -dimensional array (M components)
 * sigma -> M * D dimensional array ( each row represents diagonal coefficients for each sigma)
 * w -> M dimensional array (mixing coefficient)
 * M -> number of components
 * shared mem -> D * QT * sizeof(double)
 */
__global__ void calc_log_gamma(double * X, double * gamma_hat, double* mu,
		double* sigma, double* w, int d, int M) {
	extern __shared__ double x[];
	/* memory size should be give as 3rd parameter
	 when calling the kernel */
	int id_x = blockIdx.x;
	// Coping to Shared Memory
	if (threadIdx.x == 0){
		for (int i = 0; i < QT; ++i) {
			for (int j = 0; j < d; ++j) {
				x[i * d + j] = X[(id_x + i) * d + j];
			}
		}
	}
	__syncthreads();
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
//	printf("w det %lf %lf \n", w[m], det);
	// log(2*pi)*0.5 = 0.399089934
	double Gm = log(w[m]) + 0.399089934 * d + 0.5 * log(abs(det));
	__syncthreads();
	for (i = 0; i < QT; i++) {
		gamma_[i] = Gm;
	}
	int N = gridDim.x * QT;
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
		gamma_hat[m * N + k + blockIdx.x*QT] = gamma_[k];
//		printf("%lf %d %d %d\n", gamma_hat[m * N + k + blockIdx.x*QT], m ,
//				k + blockIdx.x*QT, m * N + k + blockIdx.x*QT);
	}
}

/* 
 * normalize gamma
 * number of blocks => S1*QT
 * number of threads per block => M
 * gamma => unnormalized responsibilities M * T array
 * log_like => T dimensional array containing sum of gamma
 */
__global__ void normilize(double * gamma, double *log_like, int M, int N) {
	int id_x = blockIdx.x * gridDim.y + blockIdx.y;
	int j = threadIdx.x;
//	printf("gamma before %lf %lf %d % d %lf\n", gamma[id_x + j * N],log_like[id_x], j ,
//			id_x, gamma[id_x + j * N] - log_like[id_x]  );
	gamma[j * N + id_x] = exp(gamma[id_x + j * N] - log_like[id_x]);
//	printf("gamma %lf %lf %d %d \n", gamma[j * N + id_x], log_like[id_x],  j , id_x);
}
/*
 * QT*S1 => Number of blocks
 * number of threads => 1
 * M => number of components
 * N => number of points
 * gamma => N length array of log likelihoods
 * gamma_hat => M * N matrix of responsibility
 */
__global__ void calc_likelihood(double * gamma, double * gamma_hat, int M,
		int N) {
	int id_x = blockIdx.y * gridDim.x + blockIdx.x;
	double total;
	if (id_x < N) {
		total = 0;
		for (int i = 0; i < M; ++i) {
//			printf(" calc %lf %d %d \n", gamma_hat[M * i + id_x], i, id_x);
			total += exp(gamma_hat[N * i + id_x]);
		}
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
 * 1 block M threads
 * eps M * D
 * eps_sq M * D
 * T number of points
 * M number of components
 * D feature dimension
 * mu M * D (OUT)
 * sigma M * D (OUT)
 * w -> mixing coefficient M dimensional (OUT)
 */

__global__ void find_mu_sigma_omega(double *c, double* eps,double* eps_sq,
		double* gamma, int T,int M, int D,
		double * mu, double *sigma, double *w){
	int component = threadIdx.x;
	double cm = c[component];
//	for (int i = 0; i < T; ++i) {
//		cm = cm + gamma[T*component + i];
//	}
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
void write_file(double* mu,double* sigma,double* w,double* respon, int N, int D, int M)
{
	FILE* f1 = fopen("parameters.txt", "w");
	check(f1, "File parameters.txt could not be created \n");
	FILE* f2 = fopen("respon.txt", "w");
	check(f2, "File respon.txt could not be opened \n");
	fprintf(f2, "Responsibility matrix (each component forms each row and each column forms each point )\n");
	for (int i = 0; i < M; ++i) {
		fprintf(f1, "mixing coefficient of component %d \n", i);
		fprintf(f1,"%lf\n" ,w[i]);
		fprintf(f1, "sigma(diagonal) of component %d \n", i);
		for (int j = 0; j < D; ++j) {
			fprintf(f1,"%lf\t" ,sigma[i*D+j]);
		}
		fprintf(f1, "\n");
		fprintf(f1, "mu of component %d \n", i);
		for (int j = 0; j < D; ++j) {
			fprintf(f1,"%lf\t" ,mu[i*D+j]);
		}
		fprintf(f1, "\n");
		for (int k = 0; k < N; ++k) {
			fprintf(f2,"%lf\t" ,respon[i*N+k]);
		}
		fprintf(f2, "\n");
	}
	fclose(f1);
	fclose(f2);
}
/*
 * Format of argv
 * <input filename>  <number of clusters>
 */
int main(int argc, char **argv) {

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
	bool end = false;
	double *d_loglike, *d_X, *d_gamma, *d_w, *d_sigma, *d_mu, *d_c, *d_eps,
			*d_eps_sq;
	double *loglike;
	double *X;
	clock_t start,stop;
	X = input.X;
	loglike = (double *)calloc(N , sizeof(double));
	check(loglike, "Unable to allocate MAIN MEMORY (RAM CPU)");
	double old_log_like = 0;
	int S1 = ceil(N * 1.0 / QT);
	int S2 = ceil(M * 1.0 / QM);
	int S3 = ceil(D * 1.0 / QD);
	//printf("%d", S1);
	int iteration = 0;
	int max_iteratation = 30;
	double threshhold = 10;
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_c, sizeof(double) * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_eps, sizeof(double) * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_eps_sq, sizeof(double) * M));
	CUDA_SAFE_CALL(cudaMalloc((void ** )&d_loglike, sizeof(double) * N));
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
	double new_log = 0;
	double * mu, *sigma, *w, *respon;
	int * pred;
	mu = (double *)calloc(M * D, sizeof(double));
	check(mu, "Unable to allocate MAIN MEMORY (CPU)");
	sigma = (double *)calloc(M * D, sizeof(double));
	check(sigma, "Unable to allocate MAIN MEMORY (CPU)");
	w = (double *)calloc(M , sizeof(double));
	check(w, "Unable to allocate MAIN MEMORY (CPU)");
	respon = (double *)calloc(M*N , sizeof(double));
	check(respon, "Unable to allocate MAIN MEMORY (CPU)");
	pred = (int *)calloc(N , sizeof(int));
	check(respon, "Unable to allocate MAIN MEMORY (CPU)");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < D; ++j) {
			mu[i] = X[(rand() % N)*D + j];
		}
	}
	for (int i = 0; i < M * D; ++i) {
		sigma[i] = rand() % 10;
	}
	for (int i = 0; i < M; ++i) {
		w[i] = 1/(double)M;
	}
	CUDA_SAFE_CALL(
				cudaMemcpy(d_mu, mu, M * D * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
				cudaMemcpy(d_sigma, sigma, M * D * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(
				cudaMemcpy(d_w, w, M * sizeof(double), cudaMemcpyHostToDevice));
	dim3 dimLike(QT, S1);
	start = clock();
	while(!(end)){
		calc_log_gamma<<<dimBlock, QM, sizeof(double) * D * QT >>>(d_X, d_gamma,
				d_mu, d_sigma, d_w, D, M);
		calc_likelihood<<<dimLike , 1 >>>(d_loglike, d_gamma, M, N);
		normilize<<<normalize_Block, M>>>(d_gamma, d_loglike, M, N);
		eps_kernal<<<dimBlock2, QM>>>(d_X, d_gamma, D, N, d_eps, d_eps_sq, d_c, M, S1);
		find_mu_sigma_omega<<< 1, M>>>(d_c, d_eps, d_eps_sq, d_gamma, N, M,
				D, d_mu, d_sigma, d_w);
		CUDA_SAFE_CALL(
					cudaMemcpy(loglike, d_loglike, N * sizeof(double), cudaMemcpyDeviceToHost));
		new_log = 0;
		for (int i = 0; i < N; ++i) {
			new_log = new_log + loglike[i];
			//printf("%lf \n ", loglike[i]);
		}
		if ((abs(new_log - old_log_like)) < threshhold){
			end = true;
		}
		if (iteration >= max_iteratation){
			end = true;
		}
		iteration ++;
		old_log_like = new_log;
		printf("iteration %d log = %lf \n", iteration, new_log);
	}
	stop = clock();
	printf("time %lf \n ", (double)(stop-start)/CLOCKS_PER_SEC);
	CUDA_SAFE_CALL(
		cudaMemcpy(w, d_w, M * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(
		cudaMemcpy(sigma, d_sigma, M * D * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(
			cudaMemcpy(mu, d_mu, M * D * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(
			cudaMemcpy(respon, d_gamma, M * N * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_loglike));
	CUDA_SAFE_CALL(cudaFree(d_X));
	CUDA_SAFE_CALL(cudaFree(d_gamma));
	CUDA_SAFE_CALL(cudaFree(d_w));
	CUDA_SAFE_CALL(cudaFree(d_mu));
	CUDA_SAFE_CALL(cudaFree(d_sigma));

	double maxi;
	for (int i = 0; i < N; ++i) {
		maxi = 0;
		for (int j = 0; j < M; ++j) {
//			printf("respon %lf %d %d \t",respon[j*N+i] , j, i);
			if (maxi < respon[j*N+i]) {
				pred[i]=j;
				maxi = respon[j*N+i];
			}
		}
		printf("\npred %d %d \n ",i,pred[i]);
	}
	write_file(mu, sigma, w, respon, N, D, M);
	return 0;
}
