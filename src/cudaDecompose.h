#ifndef CUDA_DECOMPOSE
#define CUDA_DECOMPOSE

void cpu_correlate(float *source_means, float *destin_means, float *pearsonsr);
void DecomposeAndMatch(CudaImage &img1, CudaImage &img2,
                       CudaImage &mem1, CudaImage &mem2,
                       float soriginx, float soriginy, float doriginx, float doriginy,
                       int start,
                       int *source_extent, int *destination_extent);
#endif
