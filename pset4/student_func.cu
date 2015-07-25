//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ compute_histogram(unsigned int* const input,
                             unsigned int* output,
                             unsigned int mask,
                             unsigned int iteration,
                             const size_t numElems) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numElems) {
        return;
    }
    unsigned int bin = (input[idx] & mask) >> iteration;
    atomicAdd(&(output[1-bin]), 1);
}

__global__ void compute_cumulative_hist_naive(unsigned int* d_in,
                                        unsigned int* const d_out,
                                        const size_t numBins) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBins) {
        return;
    }

    extern __shared__ unsigned int tmp[];
    int pout = 0, pin = 1;

    tmp[pout * numBins + idx] = (idx > 0)? d_in[idx - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < numBins; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (idx >= offset) {
            tmp[pout * numBins + idx] += tmp[pin * numBins + idx - offset];
        } else {
            tmp[pout * numBins + idx] += tmp[pin * numBins + idx];
        }
        __syncthreads();
    }

    d_out[idx] = tmp[pout * numBins + idx];        
}

__global__ void scatter(unsigned int* const d_inputVals,
                        unsigned int* const d_inputPos,
                        unsigned int* const d_outputVals,
                        unsigned int* const d_outputPos,
                        unsigned int* cum_hist,
                        const size_t numElems,
                        const size_t numBins,
                        unsigned int mask,
                        unsigned int iteration) {

    int idx = blockDim.x * blockIdx.idx + threadIdx.idx;
    if (idx >= numElems) {
        return;
    }
    
    unsigned int bin = (input[idx] & mask) >> iteration;
    unsigned int last_elem_bin = (input[numElems - 1] & mask) >> iteration;
    unsigned int total_zeros = cum_hist[numBins - 1] + 1 - last_elem_bin;
    unsigned int t = idx - cum_hist[idx] + total_zeros;

    unsigned int dst_pos = bin? t : cum_hist[idx];

    d_outputVals[dst_pos] = d_inputVals[idx];
    d_outputPos[dst_pos] = d_inputPos[idx];
}

__global__ void complete_one_sort(unsigned int* const d_inputVals,
                                  unsigned int* const d_inputPos,
                                  unsigned int* const d_outputVals,
                                  unsigned int* const d_outputPos,
                                  const size_t numElems,
                                  unsigned int* offset,
                                  unsigned int* cum_hist,
                                  unsigned int mask,
                                  unsigned int iteration) {

    int idx = blockDim.x * blockIdx.idx + threadIdx.idx;
    if (idx >= numElems) {
        return;
    }

    unsigned int bin = (d_inputVals[idx] & mask) >> iteration;

    unsigned int dst_pos = offset[idx]; 
    d_outputVals[dst_pos] = d_inputVals[idx];
    d_outputPos[dst_pos] = d_inputPos[idx];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    const int numBits = 1;
    const int numBins = 1 << numBits;

    // Initialize histogram and cumulative histogram memory
    unsigned int* hist;
    unsigned int* cum_hist;
    /*unsigned int* offset;*/
    checkCudaErrors(cudaMalloc(hist, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMalloc(cum_hist, sizeof(unsigned int) * numBins));
    /*checkCudaErrors(cudaMalloc(offset, sizeof(unsigned int) * numElems));*/

    // Compute block and grid size
    const dim3 hist_blockSize(256, 1, 1);
    const dim3 hist_gridSize((numElems + 256 - 1)/256, 1, 1);
    const dim3 scan_blockSize(1, 1, 1);
    const dim3 scan_gridSize(numBins, 1, 1);
    const dim3 offset_blockSize(256, 1, 1);
    const dim3 offset_gridSize((numElems + 256 - 1)/256, 1, 1);
    const dim3 sort_blockSize(256, 1, 1);
    const dim3 sort_gridSize((numElems + 256 - 1)/256, 1, 1);

    unsigned int *tmp;

    for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i+= numBits) {
        unsigned int mask = (numBins - 1) << i;
        checkCudaErrors(cudaMemset(hist, 0, sizeof(unsigned int) * numBins));
        checkCudaErrors(cudaMemset(cum_hist, 0, sizeof(unsigned int) * numBins));

        // create histogram of number of occurrences of each digit
        compute_histogram<<<hist_gridSize, hist_blockSize>>>(d_inputVals, hist, mask, i, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // exclusive prefix sum of histogram
        compute_cumulative_hist_naive<<<scan_gridSize, scan_blockSize>>>(hist, cum_hist, numBins);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // determine relative offset of each digit
        /*checkCudaErrors(cudaMemset(offset, 0, sizeof(unsigned int) * numElems));*/
        /*compute_relative_offset<<<offset_gridSize, offset_blockSize>>>(d_inputVals,*/
        /*                                                               offset,*/
        /*                                                               cum_hist,*/
        /*                                                               numElems,*/
        /*                                                               mask,*/
        /*                                                               iteration);*/
        /*cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());*/

        scatter<<<sort_gridSize, sort_blockSize>>>(d_inputVals,
                                                   d_inputPos,
                                                   d_outputVals,
                                                   d_outputPos,
                                                   cum_hist,
                                                   numElems,
                                                   numBins,
                                                   mask,
                                                   iteration);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
        // Swap input and output
        tmp = d_outputVals;
        d_outputVals = d_inputVals;
        d_inputVals = tmp;

        tmp = d_outputPos;
        d_outputPos = d_inputPos;
        d_inputPos = tmp;
    }

    // Swap input and output
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

    // Free memory
    checkCudaErrors(cudaFree(hist));
    checkCudaErrors(cudaFree(cum_hist));
}
