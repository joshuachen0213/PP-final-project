#ifndef __MATRIX_HPP
#define __MATRIX_HPP
#include <math.h>

void lu(int n, double *mat, double *lower, double *upper);
void inverse_matrix(int n, double *mat, double **inv);
void multiply(int n, double *a, double *b, double **out);

#endif