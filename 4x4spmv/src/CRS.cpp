#include "spmv.h"

void spmv(int *rowPtr, int *columnIndex,
        DTYPE *values, DTYPE *y, DTYPE *x){
    L1: for (int i = 0; i < NUM_ROWS; i++) {
        DTYPE y0 = 0;
    L2: for (int k = *(rowPtr+i); k <*(rowPtr+i+1); k++) {
            #pragma HLS unroll factor=8
            #pragma HLS pipeline
            y0 +=(*(values+k))* (*(x+(*(columnIndex+k))));
        }
        *(y+i) = y0;
    }
}

void spmv_wrap(int *rowPtr, int *columnIndex,DTYPE *values,DTYPE *y,DTYPE*x){
    #pragma HLS INTERFACE m_axi port=rowPtr offset=slave depth=99
    #pragma HLS INTERFACE m_axi port=columnIndex offset=slave depth=99
    #pragma HLS INTERFACE m_axi port=values offset=slave depth=99
    #pragma HLS INTERFACE m_axi port=y offset=slave depth=99
    #pragma HLS INTERFACE m_axi port=x offset=slave depth=99
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    spmv(rowPtr, columnIndex, values, y, x);
}
