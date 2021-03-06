/*
 * common.h
 *
 *  Created on: 17-Mar-2018
 *      Author: libin
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h> // for FILE typedef
#include <errno.h>
// Prints verbose output during the algorithm
// Enables the DEBUG macro
#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#define DEBUG(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG(fmt, ...)
#endif

#define log_err(M, ...)                                                        \
  fprintf(stderr, "[ERROR] (%s:%d: %s) " M "\n", __FILE__,       \
          __LINE__, __FUNCTION__, ##__VA_ARGS__)

#define check(A, M, ...)                                                       \
  if (!(A)) {                                                                  \
    log_err(M, ##__VA_ARGS__);                                                 \
    exit(1); 																	\
  }

#define CUDA_SAFE_CALL( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
            __FILE__, __LINE__, cudaGetErrorString( err) );                  \
             exit(EXIT_FAILURE);                                             \
    } } while (0)

#endif /* COMMON_H_ */
