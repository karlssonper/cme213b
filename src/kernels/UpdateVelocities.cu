__global__
void updateVelocities(const float * d_levelset,
                      const float * d_velIn_x,
                      const float * d_velIn_y,
                      float * d_velOut_x,
                      float * d_velOut_y,
                      const float * d_pressure)
{

}

void updateVelocities(dim3 blocks,
                      dim3 threads,
                      const float * d_levelset,
                      const float * d_velIn_x,
                      const float * d_velIn_y,
                      float * d_velOut_x,
                      float * d_velOut_y,
                      const float * d_pressure)
{
    updateVelocities<<<blocks,threads>>(d_levelset,
                                        d_velIn_x,
                                        d_velIn_y,
                                        d_velOut_x,
                                        d_velOut_y,
                                        d_pressure);
}