#include <fstream>
#include <iomanip>
#include <iostream>

#include "matrix.hpp"

int n, step, recordSteps;
double step_size, width;
double kappa;

void read_input(const char *filename, double **uinit);
void solve(double *uinit, double *result);
void write_output(const char *filename, double *result);

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <input> <output>" << std::endl;
		return 1;
	}
	double *uinit, *result;
	read_input(argv[1], &uinit);
	result = new double[n * (step + 1)];
	solve(uinit, result);
	write_output(argv[2], result);
	return 0;
}

void solve(double *uinit, double *result) {
	double *M = new double[n * n];
	double *K = new double[n * n];
	double *M_inv, *C;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				M[i * n + j] = 4.0;
				K[i * n + j] = 2.0;
			} else if (i == j + 1 || j == i + 1) {
				M[i * n + j] = 1.0;
				K[i * n + j] = -1.0;
			} else {
				M[i * n + j] = 0.0;
				K[i * n + j] = 0.0;
			}
		}
	}
	inverse_matrix(n, M, &M_inv);
	multiply(n, M_inv, K, &C);
	double temp1 = (double)(n + 1) / width;
	double temp = kappa * 6.0 * temp1 * temp1 * step_size;
	temp *= -1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			C[i * n + j] *= temp;
		}
	}
	for (int i = 0; i < n; i++) {
		result[i] = uinit[i];
	}
	for (int i = 1; i <= step; i++) {
		for (int j = 0; j < n; j++) {
			result[i * n + j] = result[(i - 1) * n + j];
			for (int k = 0; k < n; k++) {
				result[i * n + j] += C[j * n + k] * result[(i - 1) * n + k];
			}
		}
	}
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
