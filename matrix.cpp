/*
 * This file implements the matrix inverse using LU decomposition
 * The code is modified from the original code in the following link:
 * https://github.com/RMS21/matrix-inverse-decomposition-lu/tree/master
 */

#include "matrix.hpp"

void lu(int n, double *mat, double *lower, double *upper) {
	int i = 0, j = 0, k = 0, base;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			base = j * n + i;
			if (j < i)
				lower[base] = 0;
			else {
				lower[base] = mat[base];
				for (k = 0; k < i; k++) {
					lower[base] -= lower[j * n + k] * upper[k * n + i];
				}
			}
		}
		for (j = 0; j < n; j++) {
			base = i * n + j;
			if (j < i)
				upper[base] = 0;
			else if (j == i)
				upper[base] = 1;
			else {
				upper[base] = mat[base];
				for (k = 0; k < i; k++) {
					upper[base] -= lower[i * n + k] * upper[k * n + j];
				}
				upper[base] /= lower[i * n + i];
			}
		}
	}
}

double compute_z(int n, int col, int row, double *lower, double *Z, double *I) {
	double sum = 0;
	for (int i = 0; i < n; i++) {
		if (i != row) {
			sum += lower[row * n + i] * Z[i * n + col];
		}
	}

	double result = I[row * n + col] - sum;
	result = result / lower[row * n + row];

	return result;
}

double compute_inverse(
	int n, int col, int row, double *upper, double *Z, double *inverse) {
	double sum = 0;
	for (int i = 0; i < n; i++) {
		if (i != row) {
			sum += upper[row * n + i] * inverse[i * n + col];
		}
	}

	double result = Z[row * n + col] - sum;
	result = result / upper[row * n + row];

	return result;
}

void inverse_matrix(int n, double *mat, double **inv) {
	double *lower = new double[n * n];
	double *upper = new double[n * n];
	double *Z = new double[n * n];
	double *I = new double[n * n];
	*inv = new double[n * n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			Z[i * n + j] = 0;
			(*inv)[i * n + j] = 0;
			if (i == j) {
				I[i * n + j] = 1;
			} else {
				I[i * n + j] = 0;
			}
		}
	}
	lu(n, mat, lower, upper);
	// compute z
	for (int col = 0; col < n; col++) {
		for (int row = 0; row < n; row++) {
			Z[row * n + col] = compute_z(n, col, row, lower, Z, I);
		}
	}
	// compute inverse
	for (int col = 0; col < n; col++) {
		for (int row = n - 1; row >= 0; row--) {
			(*inv)[row * n + col] =
				compute_inverse(n, col, row, upper, Z, *inv);
		}
	}
	delete[] lower;
	delete[] upper;
	delete[] Z;
	delete[] I;
}

void multiply(int n, double *a, double *b, double **out) {
	*out = new double[n * n];
	double *out_ = *out;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			out_[i * n + j] = 0;
			for (int k = 0; k < n; k++) {
				out_[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

#ifdef __UNIT_TEST__
#include <iostream>
#define SMALL 0.001
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RESET "\x1B[0m"
void output(int n, double *mat) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << mat[i * n + j] << " ";
		}
		std::cout << std::endl;
	}
}
void check_identity(int n, double *mat) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				if (fabs(mat[i * n + j] - 1) > SMALL) {
					std::cerr << COLOR_RED << "Unit Test Failed" << std::endl;
					return;
				}
			} else {
				if (fabs(mat[i * n + j]) > SMALL) {
					std::cerr << COLOR_RED << "Unit Test Failed" << std::endl;
					return;
				}
			}
		}
	}
	std::cerr << COLOR_GREEN << "Unit Test Passed" << COLOR_RESET << std::endl;
}
void multiply_and_check(int n, double *a, double *b, double **out) {
	multiply(n, a, b, out);
	// output(n, *out);
	check_identity(n, *out);
}

int main() {
	// case 1
	double mat1[] = {1, 2, 3, 4};
	double *out1;
	double *inv1;
	inverse_matrix(2, mat1, &inv1);
	multiply_and_check(2, mat1, inv1, &out1);
	delete[] inv1;
	// case 2
	double mat2[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
	double *out2;
	double *inv2;
	inverse_matrix(3, mat2, &inv2);
	multiply_and_check(3, mat2, inv2, &out2);
	delete[] inv2;
	return 0;
}
#endif