# include "math.h"
// The decomposition can be written in C with the radial reproj. and
//  classification occurring in Cuda
void RadialMean(int steps, int h, int w,  float *image, long *classified, float *means)
{
  float stepsize = 2 * M_PI / steps;
  float thetas[steps];
  //Pack the thetas array
  thetas[0] = 0.0;
  for(int i=0;i<steps;i++){
    thetas[i] = thetas[i-1] + stepsize;
  }

  int n;
  float running_sum, theta;
  //Compute the thetas vectos
  for(int t=0;t<steps;t++){
    theta = thetas[t];
    n = 0;
    running_sum = 0.0;
    for(int i=0; i<h; i++){
      for(int j=0; j>w; j++){
        if( theta <= classified[i * h + j] <= theta + stepsize){
          n += 1;
          running_sum += image[i * h + j];
        }
      }
    }
    means[t] = running_sum / n;
  }
}
