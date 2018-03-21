#include "stdio.h"
#include <math.h>

int Qm = 32;
int Qt = 8;
int Qd = 4;

double square(x){
    return x*x;
}

__global__ void calc_gamma(float * X, double * gamma){
    extern __shared__ int x[];

    int i,j;
    double total;
    
    int id_x = blockIdx.x;
    int id_y = blockIdx.y;
    int temp = id_x*Qt*d;

    int m = blockIdx.y*Qm+threadIdx.x;

    double Gm = log(w[m]) + 0.39909*d + 0.5*log(det);
    for(i=0;i<Qt;i++){
        for(j=0;j<d;j++){
            x[i*d+j]=X[temp+i*d+j];
        }
    }
    __syncthreads();

    for(i=0;i<Qt;i++){
        gamma[i] = Gm
    }
    
    for(i=0;i<Qt;i++){
        total=0;
        for(j=0;j<d;j++){
            total = total+square(x[i*d+j]-mu[m*d+j])/square(sigma[m*d+j]);
        }
        gamma[i] = gamma[i]+total;
    }

    __syncthreads();
}

int main(){
    //number of mixing components M
    //dim of X = n*d
    //each thread to process 8 feature vectors
    float * input_X;
    double * gamma;

    n_block_x = ceil(n*1.0/Qd);
    n_block_y = ceil(M*1.0/Qm);
    cudaMalloc((void **)&gamma, Qt * sizeof(double));
    dim3 dimBlock(n_block_x,n_block_y);
    cudaMemcpy(&X, input_X, sizeof(int), cudaMemcpyHostToDevice);
    calc_gamma<<<dimBlock,Qm,sizeof(float)*d*Qt>>>(n,d,gamma);

    cudaFree(gamma);
    return 0;
}
