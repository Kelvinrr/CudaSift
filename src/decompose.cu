#include "math.h"
#include "stdio.h"
#include "cudautils.h"
#include "cudaImage.h"

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

__global__ void RadialMean(float *classif, int originx, int originy, int width, int height, float *thetas)
{

  __shared__ float means[720];
  __shared__ int mean_counts[720];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // Classify the pixel to some theta and collect the dn sums at each radial slice
  if (y < height && x < width)
  {
    float theta = atan2f(y - originy, x - originx);

    // To debug and visualize, set the pixel to theta and return
    //classif[offset] = theta;

    //Find the index in the means vector that the dn value should be added to.
    float dn_value = classif[offset];
    size_t index = 0;
    while (index < 720 && thetas[index] != theta){
      index++;
    }
    // Add the value to the means and increment the counts
    atomicAdd(&means[index], dn_value);
    atomicAdd(&mean_counts[index], 1);
    __syncthreads();  // wait for everyone to finish
  }

  // Now compute the means
  if (blockIdx.x == 0 && threadIdx.x < 720)
  {
    // Utilize the thetas memory to store the means out of shared GPU memory
    thetas[threadIdx.x] = means[threadIdx.x] / mean_counts[threadIdx.x];
  }
  __syncthreads();
}

//__global__ void Decompose(float *img1_data,  float *img2_data, float *mem1_data, float *mem2_data, int soriginx, int soriginy, int doriginx, int doriginy){
  // Map from thread/block to pixel position
//}

__global__ void SourceRadialClassify(float *mem, float start, int originx, int originy, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // Classify the pixel to some theta
  if (y < height && x < width)
  {
    float theta = atan2f(y - originy, x - originx);
    float min = -1.0 * M_PI;
    float step = M_PI / 2.0;
    for(int i=0;i<=4;i++){
      if(min <= theta && theta <= min + step){
        mem[offset] = i;
      }
      min += step;
    }
  }
}

__global__ void DestinRadialClassify(float *mem, float start, float rotation, int originx, int originy, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // Classify the pixel to some theta
  if (y < height && x < width)
  {
    float theta = atan2f(y - originy, x - originx);
    float lam = 0.0;

    //This is failing and returning 4 for everything...

    float min = -1.0 * M_PI;
    float step = M_PI / 2.0;
    float start_theta = min + rotation;
    float stop_theta = 0;
    float twopi = 2 * M_PI;
    
    for(int i=0;i<=4;i++){
      stop_theta = start_theta + step;

      if(stop_theta > twopi){
        stop_theta -= twopi;
      }
      if(start_theta > twopi){
        start_theta -= twopi;
      }

      if(start_theta > stop_theta){
        if (start_theta <= theta && theta <= twopi){
          mem[offset] = i;
        }
        else if(0 <= theta && theta <= stop_theta + lam){
          mem[offset] = i;
        }
        else if(start_theta <= theta && theta <= stop_theta){
          mem[offset] = i;
        }
      } else if(start_theta <= theta && theta <= stop_theta) {
        mem[offset] = i;
      }
      start_theta += step;
    }
  }
}

__global__ void correlate(float *x, float *y, float *thetas)
  {
  int tid = threadIdx.x;
  int n = 720;
  int new_idx;
  float ry[720];
  float r, xx[720], yy[720], nr=0, dr_1=0, dr_2=0, dr_3=0, dr=0;
  float sum_y = 0, sum_yy=0, sum_xy=0, sum_x=0, sum_xx=0;
  int i = 0;

  // 'Rotate' the dmeans vector
  for(i=0;i<n;i++)
  {
    new_idx = i + tid;
    if(new_idx >= n)
    {
        new_idx -= n;
    }
    ry[i] = y[new_idx];
  }

  // Compute the correlation coefficient using https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
  // This algorithm can be numerically unstable - when?
  for(i=0;i<n;i++)
  {
   xx[i]=x[i]*x[i];
   yy[i]=ry[i]*ry[i];
  }
  for(i=0;i<n;i++)
  {
   sum_x+=x[i];
   sum_y+=ry[i];
   sum_xx+= xx[i];
   sum_yy+=yy[i];
   sum_xy+= x[i]*ry[i];
  }
  nr=(n*sum_xy)-(sum_x*sum_y);
  float sum_x2=sum_x*sum_x;
  float sum_y2=sum_y*sum_y;
  dr_1=(n*sum_xx)-sum_x2;
  dr_2=(n*sum_yy)-sum_y2;
  dr_3=dr_1*dr_2;
  dr=sqrt(dr_3);
  r=(nr/dr);
  thetas[tid] = r;

  __syncthreads();
}

void DecomposeAndMatch(CudaImage &img1, CudaImage &img2, CudaImage &mem1, CudaImage &mem2, int soriginx, int soriginy, int doriginx, int doriginy, int source_extent[4], int destination_extent[4])
{
  int w1 = img1.width;
  int h1 = img1.height;
  int w2 = img2.width;
  int h2 = img2.height;

  int start = 1;  // This needs to be passed in.  This is the new partition starting number
  int steps = 720;
  float source_means[720];
  float destin_means[720];

  // Set up a thetas vector for classifying
  float stepsize = 2 * M_PI / steps;
  float thetas[steps];
  //Pack the thetas array
  thetas[0] = 0.0;
  for(int i=0;i<steps;i++){
    thetas[i] = thetas[i-1] + stepsize;
  }

  // Get thetas over to the GPU
  float *dev_thetas;
  cudaMalloc((void**)&dev_thetas, steps * sizeof(float));
  cudaMemcpy(dev_thetas, thetas, steps * sizeof(float), cudaMemcpyHostToDevice);

  //Get the means over to two images
  dim3 threadsPerBlock(32,32); // 1024 threads per block - should be programmatic since older GPUs might only have 512
  dim3 numBlocks1((w1 + (threadsPerBlock.x - 1)) / threadsPerBlock.x,
                  (h1 + (threadsPerBlock.y - 1)) / threadsPerBlock.y);

  dim3 numBlocks2((w2 + (threadsPerBlock.x - 1))/threadsPerBlock.x,
                  (h2 + (threadsPerBlock.y - 1))/threadsPerBlock.y);
  RadialMean<<<numBlocks1, threadsPerBlock>>>(img1.d_data, soriginx, soriginy, w1, h1, dev_thetas);
  cudaMemcpy(source_means, dev_thetas, steps * sizeof(float), cudaMemcpyDeviceToHost);
  RadialMean<<<numBlocks2, threadsPerBlock>>>(img2.d_data, doriginx, doriginy, w2, h2, dev_thetas);
  cudaMemcpy(destin_means, dev_thetas, steps * sizeof(float), cudaMemcpyDeviceToHost);

  // I should now have the means from both the source and the destination images for a given subimages
  // Transfer the means over to the GPU, compute Pearson's R, and find the 'best' theta
  float *dev_source_means;
  float *dev_destin_means;
  float pearsonr[steps];
  float  *dev_pearsonr;
  cudaMalloc((void**)&dev_source_means, steps * sizeof(float));
  cudaMalloc((void**)&dev_destin_means, steps * sizeof(float));
  cudaMalloc((void**)&dev_pearsonr, steps * sizeof(float));
  cudaMemcpy(dev_source_means, source_means, steps * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_destin_means, destin_means, steps * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_pearsonr, pearsonr, steps * sizeof(float), cudaMemcpyHostToDevice);

  correlate<<<1, steps>>>(dev_source_means, dev_destin_means, dev_pearsonr);

  cudaMemcpy(pearsonr, dev_pearsonr, steps * sizeof(float), cudaMemcpyDeviceToHost);

  //iterate over the pearson r and return the maximum correlation
  float maxcorr = -1;
  int theta_idx = 0;
  float rotation = 0.0;
  for(int i=0;i<steps;i++)
  {
    //printf("%f, ", pearsonr[i]);
    if(pearsonr[i] > maxcorr)
    {
      maxcorr = pearsonr[i];
      theta_idx = i;
    }
  }
  //printf("\n");
  //printf("Max Correlation: %f\n", maxcorr);
  printf("Max Correlation Index: %i\n", theta_idx);

  rotation = (2 * M_PI / steps) * theta_idx;
  printf("Rotation Angle: %f\n", rotation);
  if(rotation > M_PI){
    maxcorr = 2 * M_PI - rotation;
  }
  //Compute the breaks using the max theta and classify the images
  //Source image never rotates, so always square
  SourceRadialClassify<<<numBlocks1, threadsPerBlock>>>(mem1.d_data, start, soriginx, soriginy, w1, h1);
  //Destination image needs to be able to rotate
  DestinRadialClassify<<<numBlocks2, threadsPerBlock>>>(mem2.d_data, start, rotation, doriginx, doriginy, w2, h2);

  // Clean Up
  cudaFree(dev_thetas);
  cudaFree(dev_destin_means);
  cudaFree(dev_source_means);
}
