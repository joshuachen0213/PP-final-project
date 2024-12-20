#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include "matrix.hpp"

#define BLOCKSIZE 32
#define HIP_CHECK(expression)                                            \
	{                                                                    \
		const hipError_t status = expression;                            \
		if (status != hipSuccess) {                                      \
			std::cerr << "HIP error " << status << ": "                  \
					  << hipGetErrorString(status) << " at " << __FILE__ \
					  << ":" << __LINE__ << std::endl;                   \
		}                                                                \
	}


int n, step, recordSteps;
double step_size, width;
double kappa;

void read_input(const char *filename, double **uinit);
void M_inverse(int n, double **M_inv);
__global__ void init_C(int n, double *C, double *M_inv, double cst);
__global__ void solve(
	int n, int step, double *uinit, double *result, double *C);
void write_output(const char *filename, double *result);

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <input> <output>" << std::endl;
		return 1;
	}
	double *uinit, *result, *M_inv;
	read_input(argv[1], &uinit);

	// allocate memory for matrix C
	M_inverse(n, &M_inv);
	double *C, *M_inv_dev;
    HIP_CHECK(hipMalloc(&M_inv_dev, n * n * sizeof(double)));
	HIP_CHECK(hipMemcpy(M_inv_dev, M_inv, n * n * sizeof(double), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&C, n * n * sizeof(double)));
	double temp = (double)(n + 1) / width;
	double cst = (-6.0) * kappa * temp * temp * step_size;
	init_C<<<1, 1024>>>(n, C, M_inv_dev, cst);
    HIP_CHECK(hipGetLastError());

	// solve
	double *uinit_dev, *result_dev;
	HIP_CHECK(hipMalloc(&uinit_dev, n * sizeof(double)));
	HIP_CHECK(hipMemcpyAsync(uinit_dev, uinit, n * sizeof(double), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&result_dev, n * (step + 1) * sizeof(double)));
    
    double alpha = 1.0, beta = 0.0;
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    double *y_dev;
    HIP_CHECK(hipMalloc(&y_dev, n * sizeof(double)));
    HIP_CHECK(hipMemcpyAsync(y_dev, uinit, n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyAsync(result_dev, uinit, n * sizeof(double), hipMemcpyHostToDevice));
    for(int i = 1; i <= step; i++) {
        hipblasDgemv(handle, HIPBLAS_OP_N, n, n,
                                   &alpha, C, n, result_dev + (i - 1) * n, 1, &beta, y_dev, 1);
        HIP_CHECK(hipMemcpyAsync(result_dev + i * n, y_dev, n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemsetAsync(y_dev, 0, n * sizeof(double)));
    }
    hipblasDestroy(handle);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

	// allocate memory for host result and write output
	HIP_CHECK(hipHostMalloc(&result, n * (step + 1) * sizeof(double)));
	HIP_CHECK(hipMemcpyAsync(result,
			  result_dev,
			  n * (step + 1) * sizeof(double),
			  hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
	write_output(argv[2], result);

	// free memory
	HIP_CHECK(hipFree(C));
	HIP_CHECK(hipFree(M_inv_dev));
	HIP_CHECK(hipFree(uinit_dev));
	HIP_CHECK(hipFree(result_dev));
	HIP_CHECK(hipHostFree(result));
	return 0;
}
__global__ void init_C(int n, double *C, double *M_inv, double cst) {
	int idx = threadIdx.x;
	int stride = blockDim.x;
	__shared__ double K[BLOCKSIZE * BLOCKSIZE];
	int x_len, y_len, x, y;
    for (int i = idx; i < n * n; i += stride) {
		C[i] = 0;
	}
    __syncthreads();
	for (int i = 0; i < (n + BLOCKSIZE - 1) / BLOCKSIZE; i++) {
		for (int j = 0; j < (n + BLOCKSIZE - 1) / BLOCKSIZE; j++) {
			x_len = std::min(i * BLOCKSIZE + BLOCKSIZE, n) - i * BLOCKSIZE;
			y_len = std::min(j * BLOCKSIZE + BLOCKSIZE, n) - j * BLOCKSIZE;
			for (int k = idx; k < x_len * y_len; k += stride) {
				x = k / y_len + i * BLOCKSIZE;
				y = k % y_len + j * BLOCKSIZE;
				if (x == y) {
					K[k] = 2.0;
				} else if (x == y + 1 || y == x + 1) {
					K[k] = -1.0;
				} else {
					K[k] = 0.0;
				}
			}
			__syncthreads();
			for (int k = idx; k < x_len * y_len; k += stride) {
                for (int h = 0; h < n; h++) {
					x = k / y_len + i * BLOCKSIZE;
					y = k % y_len + j * BLOCKSIZE;
					atomicAdd(C + h * n + y, M_inv[h * n + x] * K[k]);
				}
			}
			__syncthreads();
		}
	}
	for (int i = idx; i < n * n; i += stride) {
		C[i] *= cst;
	}
    __syncthreads();
    for (int i = idx; i < n; i += stride) {
        C[i * n + i] += 1.0;
    }
}

void M_inverse(int n, double **M_inv) {
	double *M = new double[n * n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				M[i * n + j] = 4.0;
			} else if (i == j + 1 || j == i + 1) {
				M[i * n + j] = 1.0;
			} else {
				M[i * n + j] = 0.0;
			}
		}
	}
	inverse_matrix(n, M, M_inv);
	delete[] M;
}

void write_output(const char *filename, double *result) {
	std::ofstream fout(filename);
	fout << n << std::endl << step / recordSteps << std::endl << width << std::endl;
	fout << std::fixed << std::setprecision(10);
	for (int i = 0; i <= step; i += recordSteps) {
		for (int j = 0; j < n; j++) {
			fout << result[i * n + j] << " ";
		}
		fout << std::endl;
	}
}

void read_input(const char *filename, double **uinit) {
	std::ifstream fin(filename);
	fin >> n >> step >> recordSteps >> step_size >> width >> kappa;
	*uinit = new double[n];
	for (int i = 0; i < n; i++) {
		fin >> (*uinit)[i];
	}
}
