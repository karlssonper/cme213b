__global__
void addExternalForces(const float dt,
                       const float2 force,
                       const float * d_levelset,
                       const float * d_velIn_x,
                       const float * d_velIn_y,
                       float * d_velOut_x,
                       float * d_velOut_y)
{
    // Get Index
    // Notes on indexing:
    int index  = blockDim.x* (blockIdx.x + blockIdx.y*gridDim.x) +
            threadIdx.y*blockDim.x + threadIdx.x;

    // Only compute external forces for fluid voxels
 //   if (d_levelset(index))
    {
        d_velOut_x[index] = d_velIn_x[index] + dt*force.x;
        d_velOut_y[index] = d_velIn_y[index] + dt*force.y;
    }
}

void addExternalForces(dim3 blocks, dim3 threads, const float dt,
                       const float2 force,
                       const float * d_levelset,
                       const float * d_velIn_x,
                       const float * d_velIn_y,
                       float * d_velOut_x,
                       float * d_velOut_y)
{
    addExternalForces<<<blocks, threads>>>
                                         (dt, force, d_levelset,d_velIn_x,
                                          d_velIn_y,d_velOut_x,d_velOut_y);
}
