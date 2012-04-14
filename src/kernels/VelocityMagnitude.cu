__global__
void velocityMagnitude(float * blockMags,
                       const float * d_levelset,
                       const float * d_velIn_x,
                       const float * d_velIn_y)
{

}

void velocityMagnitude(dim3 blocks, dim3 threads, float * blockMags,
                       const float * d_levelset,
                       const float * d_velIn_x,
                       const float * d_velIn_y)
{
    velocityMagnitude<<<blocks,threads>>>
                                        (blockMags, d_levelset, d_velIn_x,
                                         d_velIn_y);

}
