#include "spmv.h"

const static int S = 9;

void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
          DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
{
#pragma HLS INTERFACE mode=m_axi port=x offset=slave
#pragma HLS INTERFACE mode=m_axi port=y  offset=slave
#pragma HLS INTERFACE mode=m_axi port=values  offset=slave
#pragma HLS INTERFACE mode=m_axi port=columnIndex  offset=slave
#pragma HLS INTERFACE mode=m_axi port=rowPtr  offset=slave
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
  L1: for (int i = 0; i < NUM_ROWS; i++) {
      DTYPE y0 = 0;
    L2_1: for (int k = rowPtr[i]; k < rowPtr[i+1]; k += S) {
#pragma HLS pipeline II=S
          DTYPE yt = values[k] * x[columnIndex[k]];
      L2_2: for(int j = 1; j < S; j++) {
              if(k+j < rowPtr[i+1]) {
                  yt += values[k+j] * x[columnIndex[k+j]];
              }
          }
          y0 += yt;
      }
    y[i] = y0;
  }
}
