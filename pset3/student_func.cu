/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them. Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void get_logLum_max_min(const float* const d_logLuminance,
                       float* logLumArr, 
                       const int min_or_max,
                       const size_t size) {
    // looks correct
    extern __shared__ float sdata[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;

    if (idx >= size) {
        return;
    }

    sdata[thread_id] = d_logLuminance[idx];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s && (idx + s) < size && (thread_id + s) < blockDim.x) {
            float a = sdata[thread_id];
            float b = sdata[thread_id + s];
            if (min_or_max == 1 && a < b) {
                sdata[thread_id] = b;
            } else if (min_or_max == 0 && a > b) {
                sdata[thread_id] = b;
            }
        } 
        __syncthreads();
    }
   
    if (thread_id == 0) {
        logLumArr[blockIdx.x] = sdata[0];
    }
}

float helper_compute_min_max(const float* const d_logLuminance, 
                            const int min_or_max,
                            const size_t numRows,
                            const size_t numCols) {
    // looks correct

    size_t size = numRows * numCols;
    const int reduce_blockSize = 512;
    int reduce_gridSize;
    float* d_in;
    float* d_out;

    checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * size));
    checkCudaErrors(cudaMemcpy(d_in, d_logLuminance, sizeof(float) * size, cudaMemcpyDeviceToDevice));

    while (size > 1) {
        reduce_gridSize = (size + reduce_blockSize - 1) / reduce_blockSize;

        // Allocate memory for logLum
        float* d_out;
        checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * reduce_gridSize));

        // Find the minimum and maximum across the image (reduce)
        get_logLum_max_min<<<reduce_gridSize, reduce_blockSize, reduce_blockSize * sizeof(float)>>>(d_in,
                                                                                                    d_out,
                                                                                                    min_or_max,
                                                                                                    size);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
        // Free memory
        checkCudaErrors(cudaFree(d_in));

        d_in = d_out;
        size = reduce_gridSize;
    }

    float result;
    checkCudaErrors(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free memory
    checkCudaErrors(cudaFree(d_out));

    return result;
}

__global__ void compute_histogram(const float* const d_logLuminance, 
                       float min_logLum,
                       float max_logLum,
                       const size_t numRows,
                       const size_t numCols,
                       const size_t numBins, 
                       unsigned int* logLum_hist) {
    // Looks correct

    const int2 thread_2D_pos = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                        blockDim.y * blockIdx.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) {
        return;
    }
    
    float luminance = d_logLuminance[thread_1D_pos];
    float logLumRange = max_logLum - min_logLum;
    unsigned int bin = static_cast<unsigned int>((luminance - min_logLum) / logLumRange * numBins);
    if (bin > static_cast<unsigned int>(numBins - 1)) {
        bin = static_cast<unsigned int>(numBins - 1);
    }
    atomicAdd(&(logLum_hist[bin]), 1);
}

__global__ void compute_cumulative_hist(unsigned int* logLum_hist,
                                        unsigned int* const d_cdf,
                                        const size_t numBins) {
    // Perform exclusive scan (Blelloch scan)
    // Assume numBins is a multiple of two
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (2 * idx >= numBins || (2 * idx + 1) >= numBins) {
        return;
    }

    extern __shared__ unsigned int tmp[];
    tmp[2 * idx] = logLum_hist[2 * idx];
    tmp[2 * idx + 1] = logLum_hist[2 * idx + 1];
    int offset = 1;

    // Perform upsweep
    for (int d = numBins >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (idx < d) {
            int index1 = offset * (2 * idx + 1) - 1;
            int index2 = offset * (2 * idx + 2) - 1;
            tmp[index2] += tmp[index1];
        }
        offset *= 2;
    }

    // Perform downsweep
    if (idx == 0) {
        tmp[numBins - 1] = 0;
    }
    for (int d = 1; d < numBins; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            int index1 = offset * (2 * idx + 1) - 1;
            int index2 = offset * (2 * idx + 2) - 1;

            unsigned int t = tmp[index1];
            tmp[index1] = tmp[index2];
            tmp[index2] += t;
        }
    }

    d_cdf[2 * idx] = tmp[2 * idx];
    d_cdf[2 * idx + 1] = tmp[2 * idx + 1];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    // Set block size and grid size
    const dim3 hist_blockSize(32, 32, 1);
    const dim3 hist_gridSize((numCols + 32 - 1)/32, (numRows + 32 - 1) / 32, 1);

    const dim3 scan_blockSize(256, 1, 1);
    const dim3 scan_gridSize((numBins/2 + 256 - 1)/256, 1, 1);

    // Compute minimum and maximum
    max_logLum = helper_compute_min_max(d_logLuminance, 1, numRows, numCols);
    min_logLum = helper_compute_min_max(d_logLuminance, 0, numRows, numCols);

    // Allocate memory for histogram
    unsigned int* logLum_hist;
    checkCudaErrors(cudaMalloc(&logLum_hist, sizeof(unsigned int) * numBins));

    // Build a histogram (atomicAdd)
    compute_histogram<<<hist_gridSize, hist_blockSize>>>(d_logLuminance, 
                                               min_logLum,
                                               max_logLum, 
                                               numRows, 
                                               numCols,
                                               numBins,
                                               logLum_hist);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Cumulative add (scan)
    compute_cumulative_hist<<<scan_blockSize, scan_gridSize, sizeof(unsigned int) * numBins>>>(logLum_hist,
                                                               d_cdf,
                                                               numBins);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Free memory
    checkCudaErrors(cudaFree(logLum_hist));
}
