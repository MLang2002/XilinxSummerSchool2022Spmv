#ifndef __SPMV_H__
#define __SPMV_H__

#define SIZE   4 // SIZE of square matrix
#define NNZ   9//Number of non-zero elements
#define NUM_ROWS   4// SIZE;
typedef float DTYPE;
void spmv(int *rowPtr, int *columnIndex,DTYPE *values, DTYPE *y, DTYPE *x);
void spmv_wrap(int *rowPtr, int *columnIndex,DTYPE *values,DTYPE *y,DTYPE*x);
#endif // __MATRIXMUL_H__ not defined
