__global__
void advectVelocities(const float dt,
                      const float * d_levelset,
                      const float * d_velIn_x,
                      const float * d_velIn_y,
                      float * d_velOut_x,
                      float * d_velOut_y)
{

}

void advectVelocities(dim3 blocks, dim3 threads,const float dt,
                      const float * d_levelset,
                      const float * d_velIn_x,
                      const float * d_velIn_y,
                      float * d_velOut_x,
                      float * d_velOut_y)
{
    advectVelocities<<<blocks, threads>>>
                                        (dt, d_levelset,d_velIn_x,
                                         d_velIn_y,d_velOut_x,
                                         d_velOut_y);
}
