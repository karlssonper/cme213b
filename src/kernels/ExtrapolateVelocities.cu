__global__
void extrapolateVelocities(const float * d_levelset,
                           const float2 * d_surfacePoints,
                           const float * d_velIn_x,
                           const float * d_velIn_y,
                           float * d_velOut_x,
                           float * d_velOut_y)
{

}

void extrapolateVelocities(dim3 blocks, dim3 threads, const float * d_levelset,
                           const float2 * d_surfacePoints,
                           const float * d_velIn_x,
                           const float * d_velIn_y,
                           float * d_velOut_x,
                           float * d_velOut_y)
{
    extrapolateVelocities<<<blocks, threads>>>
                                             (d_levelset, d_surfacePoints,
                                              d_velIn_x, d_velIn_y, d_velOut_x,
                                              d_velOut_y);
}
