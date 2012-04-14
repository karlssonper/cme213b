__global__
void writePBO(uchar4 * d_pbo, const float * d_levelset)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = i + j * blockDim.x * gridDim.x;
    
    d_pbo[idx].x = 255;
    d_pbo[idx].y = 0;
    d_pbo[idx].z = 0;
    d_pbo[idx].w = 255;
}

void writePBO(dim3 blocks, dim3 threads, uchar* d_pbo, const float * d_levelset)
{
    writePBO<<<blocks,threads>>>(d_pbo,d_levelset);
}