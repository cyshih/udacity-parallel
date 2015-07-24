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
#include <stdint.h>

__global__ void get_logLum_max_min(const float* const d_logLuminance,
                       unsigned int max_logLumInt, 
                       unsigned int  min_logLumInt,
                       const size_t numRows,
                       const size_t numCols) {
    extern __shared__ float sdata_max[];
    extern __shared__ float sdata_min[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;

    if (idx >= numRows * numCols) {
        return;
    }

    sdata_max[thread_id] = d_logLuminance[idx];
    sdata_min[thread_id] = d_logLuminance[idx];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s && (idx + s) < numRows * numCols) {
            float a_max = sdata_max[thread_id];
            float b_max = sdata_max[thread_id + s];
            float a_min = sdata_min[thread_id];
            float b_min = sdata_min[thread_id + s];
            if (a_max < b_max) {
                sdata_max[thread_id] = b_max;
            } 
            if (a_min > b_min) {
                sdata_min[thread_id] = b_min;
            }
        } 
        __syncthreads();
    }
   
    if (thread_id == 0) {
        // convert float to unsigned int
        unsigned long int mask = -(sdata_max[0] >> 31) | 0x80000000;
        unsigned int max_curr = sdata_max[0] ^ mask;
        mask = -static_cast<signed long int>(sdata_min[0] >> 31) | 0x80000000;
        unsigned int min_curr = sdata_min[0] ^ mask;
        atomicMax(&max_logLumInt, max_curr);
        atomicMin(&min_logLumInt, min_curr);
    }
}

__global__ void compute_histogram(const float* const d_logLuminance, 
                       float min_logLum,
                       float max_logLum,
                       const size_t numRows,
                       const size_t numCols,
                       const size_t numBins, 
                       unsigned int* logLum_hist) {

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
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBins) {
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
    const int reduce_blockSize = 256;
    const int reduce_gridSize = (numRows * numCols + 256 - 1) / reduce_blockSize;

    const dim3 hist_blockSize(32, 32, 1);
    const dim3 hist_gridSize((numCols + 32 - 1)/32, (numRows + 32 - 1) / 32, 1);

    const dim3 scan_blockSize(256, 1, 1);
    const dim3 scan_gridSize((numBins/2 + 256 - 1)/256, 1, 1);

    // Declare and convert min and max to unsigned int
    unsigned long int mask  = -static_cast<signed long int>(max_logLum >> 31) | 0x80000000;
    unsigned int max_logLumInt = mask ^ max_logLum;
    mask = -static_cast<signed long int>(min_logLum >> 31) | 0x80000000;
    unsigned int min_logLumInt = mask ^ min_logLum;

    // Find the minimum and maximum across the image (reduce)
    get_logLum_max_min<<<reduce_gridSize, reduce_blockSize, 2 * reduce_blockSize * sizeof(float)>>>(d_logLuminance, 
                                                                                                &max_logLumInt, 
                                                                                                &min_logLumInt,
                                                                                                numRows, 
                                                                                                numCols);
    // Convert back to float
    mask = ((max_logLumInt >> 31) - 1) | 0x80000000;
    max_logLum = mask ^ max_logLumInt;
    mask = ((min_logLumInt >> 31) - 1) | 0x80000000;
    min_logLum = mask ^ min_logLumInt;

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

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
}
