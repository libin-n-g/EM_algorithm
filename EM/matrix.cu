/*
 * matrix.cu
 *
 *  Created on: 17-Mar-2018
 *      Author: libin
 */


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "matrix.cuh"
#include "common.h"

// Max Dimension of input square matrix
#define N 4
__host__ __device__ float determinant(float *mat, int n){
	float L[N][N];
	float Det = 1;
	float z = 0;
	//initIdentityGPU(L,N);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i==j) {
				L[i][j]=1;
			}
			else {
				L[i][j]=0;
			}
		}
	}
	DEBUG("Initalisation of L completed \n");
	for(int i =0;i<n;i++){
		Det = Det * mat[i*n + i];
		for(int j=i+1;j<n;j++){
			z = mat[j*n + i]/mat[i*n + i];
			for(int k=i+1; k<n;k++){
				mat[n*j+k] = mat[n*j + k] - z*mat[i*n + k];
			}
			z = L[j][i]/mat[i*n + i];
			for(int k=0; k<(i-1);k++){
				L[j][k] = Det *(L[j][k]	- z * L[i][k]);
			}
		}
	}
	return Det;
}

/*
 * Initalises n*n matrix to Idendity
 */
//__device__ void initIdentityGPU(float devMatrix[N][N], int n) {
//    int x = blockDim.x*blockIdx.x + threadIdx.x;
//    int y = blockDim.y*blockIdx.y + threadIdx.y;
//    int index = x * n + y;
//    if(y < n && x < n) {
//          if(x == y)
//              devMatrix[x][y] = 1;
//          else
//              devMatrix[x][y] = 0;
//    }
//}

// Function to get cofactor of mat[p][q] in temp[][]. n is current
// dimension of mat[][]
__global__ void getCofactor(float *mat, float *temp, int p, int q, int n)
{
//    int i = 0, j = 0;
	int row = blockIdx.x;
	int col = threadIdx.x;
 	int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Looping for each element of the matrix
    //for (int row = 0; row < n; row++)
    //{
      //  for (int col = 0; col < n; col++)
        //{
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q)
            {
            	if (row > p)
            		row = row - 1;
            	if (col > q)
            		col = col -1;
                temp[row *n + col] = mat[index];
 		//		j++;
                // Row is filled, so increase row index and
                // reset col index
      //          if (j == n - 1)
          //      {
            //        j = 0;
              //      i++;
                }
          //  }
        //}
    //}
}

/* Recursive function for finding determinant of matrix.
   n is current dimension of mat[][].
   This code can be parallelised by removing the loops
   */
float determinantOfMatrix(float *mat,  int n)
{
    float D = 0; // Initialize result

    //  Base case : if matrix contains single element
    if (n == 1)
        return mat[0];

    float *d_temp, *d_M; // To store cofactors
    float *temp;
    printf("%f\n", mat[3]);
    temp = (float *)malloc(N*N*sizeof(float));
	cudaMalloc((void **)&d_M, N*N*sizeof(float));
 	cudaMalloc((void **)&d_temp, N*N*sizeof(float));
 	cudaMemcpy(d_M, mat, N*N*sizeof(float), cudaMemcpyHostToDevice);
    int sign = 1;  // To store sign multiplier

     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of mat[0][f]
        getCofactor<<< n, n>>>(d_M, d_temp, 0, f, n);
        cudaMemcpy(temp, d_temp, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        printf("temp %f\n", temp[0]);
        D += sign * mat[f] * determinantOfMatrix(temp, n - 1);
 		printf("%f %f\n", D, mat[f]);
        // terms are to be added with alternate sign
        sign = -sign;
    }
 	cudaFree(d_temp);
 	cudaFree(d_M);
 	free(temp);
    return D;
}


/*
*********************************************************************
function name: matrix_add

description: adding of two matrix

parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a m X n matrix (B)
            &c GPU device output purpose pointer to a m X n matrix (C)
            to store the result

Note:
    grid and block should be configured as:
        	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    		unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    		dim3 dimGrid(grid_cols, grid_rows);
    		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    call the function
		matrix_add<<< dimGrid, dimBlock >>>(d_a, d_b, d_c, n, m);

*********************************************************************
*/
__global__ void matrix_add(float *a, float *b, float *c, int n, int m)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * n + col;
	if (col < n && row < m)
		c[index] = a[index] + b[index];

}

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C)
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters:
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C)
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters:
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU,
             for validating GPU results

parameters:
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C)
            to store the result
return: none
*********************************************************************
*/
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}


