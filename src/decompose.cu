#include "math.h"
#include "stdio.h"
#include "cudautils.h"
#include "cudaImage.h"

// The decomposition can be written in C with the radial reproj. and
//  classification occurring in Cuda
void RadialMean(int steps, int h, int w,  float *image, float *classified, float *means)
{
  float stepsize = 2 * M_PI / steps;
  float thetas[steps];
  //Pack the thetas array
  thetas[0] = 0.0;
  for(int i=0;i<steps;i++){
    thetas[i] = thetas[i-1] + stepsize;
  }
  int i, j;
  float running_sum, theta, n;
  //Compute the thetas vectors
  for(int t=0;t<steps;t++){  // Here is where I can parallelize this
    theta = thetas[t];
    n = 0;
    running_sum = 0.0;
    for(i=0; i<h; i++){
      for(j=0; j<w; j++){
        //printf("%.6f, %.6f, %.6f\n", theta, classified[i*h +j], theta + stepsize);
        if( theta <= classified[i * h + j] && classified[i*h+j] <= theta + stepsize)
        {
          n += 1;
          running_sum += image[i * h + j];
        }
      }
    }
    if(n != 0){
        *(means+t) = running_sum / n;
    }
  }
}

__global__ void RadialClassify(float *classif, int originx, int originy, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // Check that the thread is within the image
  if (y < height && x < width)
  {
    float theta = atan2f(y - originy, x - originx);
    classif[offset] = theta;
  }


}

//__global__ void Decompose(float *img1_data,  float *img2_data, float *mem1_data, float *mem2_data, int soriginx, int soriginy, int doriginx, int doriginy){
  // Map from thread/block to pixel position
//}

void DecomposeAndMatch(CudaImage &img1, CudaImage &img2, CudaImage &mem1, CudaImage &mem2, int soriginx, int soriginy, int doriginx, int doriginy, int source_extent[4], int destination_extent[4])
{
  int w1 = img1.width;
  int h1 = img1.height;
  int w2 = img2.width;
  int h2 = img2.height;


  dim3 threadsPerBlock(32,32); // 1024 threads per block - should be programmatic since older GPUs might only have 512
  dim3 numBlocks1((w1 + (threadsPerBlock.x - 1)) / threadsPerBlock.x,
                  (h1 + (threadsPerBlock.y - 1)) / threadsPerBlock.y);

  dim3 numBlocks2((w2 + (threadsPerBlock.x - 1))/threadsPerBlock.x,
                  (h2 + (threadsPerBlock.y - 1))/threadsPerBlock.y);
  RadialClassify<<<numBlocks1, threadsPerBlock>>>(mem1.d_data, soriginx, soriginy, w1, h1);
  RadialClassify<<<numBlocks2, threadsPerBlock>>>(mem2.d_data, doriginx, doriginy, w2, h2);


  //yblocks = h1 / dimBlock.y+((h1%dimBlock.y)==0?0:1);
  //xblocks = w1 / dimBlock.x+((w1%dimBlock.x)==0?0:1);
  //dimGrid = dim3[xblocks, yblocks;
  //RadialClassify<<<grid, 128>>>(mem2.d_data, doriginx, doriginy, w2, h2);
  //Decompose<<<grid, 1>>>(img1.d_data, img2.d_data, mem1.d_data, mem2.d_data, soriginx, soriginy, doriginx, doriginy);

}
