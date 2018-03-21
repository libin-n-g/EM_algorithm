LIBS=-lafcuda
LIB_PATHS=-L/usr/local/lib
INCLUDES=-I/usr/local/include
NVCC=nvcc # $(COMPILER_OPTIONS)
# COMPILER_OPTIONS=-std=c++11 -g
LD_LIBRARY_PATH=/usr/local/lib

all: gmmEM.cu Makefile
	$(NVCC) gmmEM.cu matrix.cu $(INCLUDES) $(LIBS) $(LIB_PATHS)