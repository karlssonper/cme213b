#include <vector_types.h>
#include <iostream>

__global__
void writePBO(uchar4 * d_pbo, const float * d_levelset)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = i + j * blockDim.x * gridDim.x;
    
    d_pbo[idx].x = 0;
    d_pbo[idx].y = 0;
    d_pbo[idx].z = d_levelset[idx] * 10;
    /*if (d_levelset[idx] <= 0)
        d_pbo[idx].z = 255;
    else
        d_pbo[idx].z = 0;*/
    
    d_pbo[idx].w = 255;
}

void writePBO(dim3 blocks, 
              dim3 threads, 
              uchar4 * d_pbo, 
              const float * d_levelset)
{
    std::cout << "Writing to PBO (blocks: "<< blocks.x << "x" << blocks.y << 
            " threads: "<< threads.x << "x" << threads.y<< ")" << std::endl;
        
    writePBO<<<blocks,threads>>>(d_pbo, d_levelset);
}