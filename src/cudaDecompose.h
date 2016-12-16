#ifndef CUDA_DECOMPOSE
#define CUDA_DECOMPOSE

void RadialMean(int steps, int h, int w,  float *image, float *classified, float *means);
void DecomposeAndMatch(CudaImage &img1, CudaImage &img2, CudaImage &mem1, CudaImage &mem2, int soriginx, int soriginy, int doriginx, int doriginy, int source_extent[4], int destination_extent[4]);
#endif
