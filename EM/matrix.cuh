/*
 * matrix.cuh
 *
 *  Created on: 17-Mar-2018
 *      Author: libin
 */

#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#define BLOCK_SIZE 16
__host__ __device__ float determinant(float *mat, int n);
float determinantOfMatrix(float *mat,  int n);
__device__ void initIdentityGPU(float *devMatrix, int n);
__global__ void matrix_add(float *a, float *b, float *c, int n, int m);
__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k);
__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n);
__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols);

#endif /* MATRIX_CUH_ */
