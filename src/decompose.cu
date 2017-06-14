#include "math.h"
#include "stdio.h"
#include "float.h"
#include "cudautils.h"
#include "cudaImage.h"


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ inline void atomicFloatAdd(float *address, float val)
{
  //From: https://devtalk.nvidia.com/default/topic/391295/atomic-float-operations-especially-add/
       int i_val = __float_as_int(val);
       int tmp0 = 0;
       int tmp1;

       while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
       {
               tmp0 = tmp1;
               i_val = __float_as_int(val + __int_as_float(tmp1));
       }
}

struct RadialCorrelation{
  int thetaidx;
  float corrcoeff;
};

__global__ void RadialMean(float *img, float originx, float originy, int width, int height, float *thetas, int *extents, float *sums, unsigned int *mean_counts)
{
  int index = 0;
  float cur = 0;
  float nearest = FLT_MAX;
  float theta;
  int nbins = 720;

  // x,y pixel position and position in the linear memory
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset =  y * width + x;

  // position in the block local memory
  int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
  int nt = blockDim.x * blockDim.y;
  int g = blockIdx.x + blockIdx.y * gridDim.x;
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  // initialize temporary accumulation arrays in global memory
  for (int i = t; i < nbins;i+=nt){
    sums[i + g * 720] = 0.0;
    mean_counts[i+g*720] = 0;
  }
  // Classify the pixel to some theta and collect the dn sums at each radial slice

  if(y >= extents[0] && y < extents[1] && x >= extents[2] && x < extents[3])
  {
  for (int col = x; col < width;col+=nx){
    for (int row = y; row < height; row += ny){
      theta = atan2f(row - originy, col - originx) + M_PI;
       //Find the nearest discrete theta
      for(int i=0;i<nbins;i++){
        cur = abs(thetas[i] - theta);
        if(cur < nearest){
          nearest = cur;
          index = i;
        }
       }
       atomicAdd(&sums[index + g * 720], img[offset]);
       atomicAdd(&mean_counts[index + g * 720], 1);
     }
    }
  }
}

__global__ void RadialMeanAccum(unsigned int *counts, float *sums, int total_blocks, float *means)
{
  int i = blockIdx.x;
  unsigned int count = 0;
  float sum = 0;
  // Accumulate the histograms
  for(int j=0;j<total_blocks;j++){
    count += counts[i + 720 * j];
    sum += sums[i + 720 * j];
  }
  means[i] = sum / count;
}

__global__ void RadialClassify(float *img, float start, float rotation, float originx, float originy, int width, int height, int *extents)
{
  // Compute the angle of rotation from some origin and
  //classify the pixels into some quadrant
  // x,y pixel position and position in the linear memory
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset =  y * width + x;

  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  float theta;

  if ((y < extents[0] || y >= extents[1])){
    return;
  }
  if ((x < extents[2] || x >= extents[3])){
    return;
  }


  theta = atan2f(y- originy, x - originx) + M_PI;

  //Classify based on rotated quadrant
  if(theta <= rotation + M_PI/2){
    img[offset] = start;
  }
  else if((theta > rotation + M_PI/2) && (theta <= rotation + M_PI)){
    img[offset] = start + 1;
  }
  else if((theta > rotation + M_PI) && (theta <= rotation + 3*M_PI/2)){
    img[offset] = start + 2;
  }
  else{
    img[offset] = start + 3;
  }
img[offset] = 10;


}

RadialCorrelation RadialCorrelate(float *x, float *y){
  int n = 720;
  int new_idx=0;
  float best = -INFINITY;
  float rotation;
  float xx[720], ry[720];
  float r, nr=0, dr_1=0, dr_2=0, dr_3=0, dr=0, sum_x2=0, sum_y2;
  double sum_y, sum_yy, sum_xy, sum_x, sum_xx=0;

  for(int i=0;i<n;i++){
    xx[i] = x[i] * x[i];
    sum_xx += xx[i];
  }

  for(int j=0;j<n;j++){
    // Rest for next rotation
    sum_x = 0;
    sum_y = 0;
    sum_yy = 0;
    sum_xy=0;

    //rotate one vector around the other
    for(int i=0;i<n;i++){
      new_idx = i+j;
      if(new_idx >= n){
        new_idx = i + j - n;
      }
      ry[i] = y[new_idx];
    }
    if(j == 1){
      for(int k=0;k<n;k++){
        printf("%f,", ry[k]);
      }
    }
    // summations for correlation coeff.
    for(int i=0;i<n;i++){
      sum_x += x[i];
      sum_y += ry[i];
      sum_yy += ry[i] * ry[i];
      sum_xy += x[i] * ry[i];
    }
    // compute correlation coeff (Pearson's R)
    nr = (n * sum_xy) - (sum_x * sum_y);
    sum_x2 = sum_x * sum_x;
    sum_y2 = sum_y * sum_y;
    dr_1=(n*sum_xx)-sum_x2;
    dr_2=(n*sum_yy)-sum_y2;
    dr_3=dr_1*dr_2;
    dr=sqrt(dr_3);
    r=(nr/dr);
    if(j<5){
      printf("\n%i, %f\n", j, r);
      printf("%f, %f\n", x[j], xx[j]);
      printf("%f, %f\n", y[j], ry[j]);
      printf("%f, %f, %f, %f, %f\n", sum_x, sum_y, sum_yy, sum_xx, sum_xy);
      printf("%f, %f, %f, %f, %f, %f, %f\n", nr, sum_x2, sum_y2, dr_1, dr_2, dr_3, dr);
    }
    if(r > best){
      best = r;
      rotation = j;
    }
  }
  struct RadialCorrelation res;
  res.thetaidx = rotation;
  res.corrcoeff = best;
  return res;
}

void DecomposeAndMatch(CudaImage &img1, CudaImage &img2, CudaImage &mem1, CudaImage &mem2, float soriginx, float soriginy, float doriginx, float doriginy, int start, int *source_extent, int *destination_extent)
{
  int w1 = img1.width;
  int h1 = img1.height;
  int w2 = img2.width;
  int h2 = img2.height;

  int steps = 720;
  size_t bytes = steps * sizeof(float);
  size_t ibytes = 4 * sizeof(int);

  int *dev_source_extent;
  int *dev_destin_extent;

  HANDLE_ERROR(cudaMalloc(&dev_source_extent, ibytes));
  HANDLE_ERROR(cudaMalloc(&dev_destin_extent, ibytes));
  HANDLE_ERROR(cudaMemcpy(dev_source_extent, source_extent, ibytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_destin_extent, destination_extent, ibytes, cudaMemcpyHostToDevice));

  // Set up a thetas vector for classifying
  float stepsize = 2 * M_PI / steps;
  float thetas[steps];
  //Pack the thetas array
  thetas[0] = 0.0;
  for(int i=0;i<steps;i++){
    thetas[i] = thetas[i-1] + stepsize;
  }
  // Get thetas over to the GPU and set the means to zeros
  float *dev_thetas;
  HANDLE_ERROR(cudaMalloc(&dev_thetas, bytes));
  HANDLE_ERROR(cudaMemcpy(dev_thetas, thetas, bytes, cudaMemcpyHostToDevice));

  //Get the means over to two images
  dim3 blockSize(32,32);
  dim3 gridSizeSource((w1 + blockSize.x - 1) / blockSize.x,
                      (h1 + blockSize.y - 1) / blockSize.y);

  int total_blocks = gridSizeSource.x * gridSizeSource.y;

  unsigned int *ds_part_counts;
  float *ds_part_sums;
  HANDLE_ERROR(cudaMalloc(&ds_part_counts, total_blocks * 720 * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMalloc(&ds_part_sums, total_blocks * 720 * sizeof(float)));

  // Allocate the source mean vector on the CPU/GPU
  float source_means[720];
  float *dev_source_means;
  HANDLE_ERROR(cudaMalloc(&dev_source_means, bytes));
  HANDLE_ERROR(cudaMemcpy(dev_source_means, source_means, bytes, cudaMemcpyHostToDevice));
  // Compute the source radial mean
  RadialMean<<<gridSizeSource, blockSize>>>(img1.d_data, soriginx, soriginy, w1, h1, dev_thetas, dev_source_extent, ds_part_sums, ds_part_counts);
  HANDLE_ERROR(cudaPeekAtLastError());
  RadialMeanAccum<<<720, 1>>>(ds_part_counts, ds_part_sums, total_blocks, dev_source_means);
  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaMemcpy(source_means, dev_source_means, bytes, cudaMemcpyDeviceToHost));

  // Working to here.
  dim3 gridSizeDestin((w2 + (blockSize.x - 1)) / blockSize.x,
                      (h2 + (blockSize.y - 1)) / blockSize.y);

  total_blocks = gridSizeDestin.x * gridSizeDestin.y;
  // Allocate the destination mean vector to the CPU/GPU
  float destin_means[720];
  float *dev_destin_means;

  HANDLE_ERROR(cudaFree(ds_part_sums));
  HANDLE_ERROR(cudaFree(ds_part_counts));

  unsigned int *dd_part_counts;
  float *dd_part_sums;
  HANDLE_ERROR(cudaMalloc(&dd_part_counts, total_blocks * 720 * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMalloc(&dd_part_sums, total_blocks * 720 * sizeof(float)));

  HANDLE_ERROR(cudaMalloc(&dev_destin_means, bytes));
  RadialMean<<<gridSizeDestin, blockSize>>>(img2.d_data, doriginx, doriginy, w2, h2, dev_thetas, dev_destin_extent, dd_part_sums, dd_part_counts);
  HANDLE_ERROR(cudaPeekAtLastError());
  RadialMeanAccum<<<720,1>>>(dd_part_counts, dd_part_sums, total_blocks, dev_destin_means);
  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaMemcpy(destin_means, dev_destin_means, bytes, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(dd_part_sums));
  HANDLE_ERROR(cudaFree(dd_part_counts));

  // Compute the rotation
  RadialCorrelation corr = RadialCorrelate(source_means, destin_means);
  float rotation = thetas[corr.thetaidx];
  printf("\n");
  printf("Max Correlation: %f\n", corr.corrcoeff);
  printf("Max Correlation Index: %i\n", corr.thetaidx);
  printf("Rotation Angle (Radians): %f\n", rotation);


  // Classfiy the rasters

  //Compute the breaks using the max theta and classify the images
  RadialClassify<<<gridSizeSource, blockSize>>>(mem1.d_data, start, 0, soriginx, soriginy, w1, h1, dev_source_extent);
  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  //printf("Start: %i\n", start);
  printf("%i, %i, %i, %i\n", source_extent[0],source_extent[1],source_extent[2],source_extent[3]);
  printf("doriginx: %f, doriginy: %f\n",doriginx, doriginy );
  //Destination image needs to be able to rotate
  RadialClassify<<<gridSizeDestin, blockSize>>>(mem2.d_data, start, rotation, doriginx, doriginy, w2, h2, dev_destin_extent);
  cudaDeviceSynchronize();
  // Clean Up
  cudaFree(dev_source_extent);
  cudaFree(dev_destin_extent);
  cudaFree(dev_thetas);
  cudaFree(dev_destin_means);
  cudaFree(dev_source_means);
}
