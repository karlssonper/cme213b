__global__
void buildLevelSetSphere(const float r,
                         const float2 center,
                         const float dx,
                         float * d_levelset)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = i + j * blockDim.x * gridDim.x;

    const float x = dx * i - center.x;
    const float y = dx * j - center.y;
    d_levelset[idx] = sqrt(x*x+y*y) - r;
}

void buildLevelSetSphere(dim3 blocks,
                         dim3 threads,
                         const float r,
                         const float2 center,
                         const float dx,
                         float * d_levelset)
{
    buildLevelSetSphere<<<blocks,threads>>>(r, center, dx, d_levelset);
}