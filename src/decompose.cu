#include "math.h"
#include "stdio.h"

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
  //Compute the thetas vectos
  for(int t=0;t<steps;t++){
    theta = thetas[t];
    n = 0;
    running_sum = 0.0;
    for(i=0; i<h; i++){
      for(j=0; j<w; j++){
        //printf("%.6f, %.6f, %.6f\n", theta, classified[i*h +j], theta + stepsize);
        if( theta <= classified[i * h + j] && classified[i*h+j] <= theta + stepsize){
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
