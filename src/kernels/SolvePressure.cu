__global__
void solvePressure(const float volumeLoss,
                   const float * d_levelset,
                   const float * d_velIn_x,
                   const float * d_velIn_y,
                   const float * d_pressureIn,
                   float * d_pressureOut)
{

}

void solvePressure(dim3 blocks,
                   dim3 threads,
                   const float volumeLoss,
                   const float * d_levelset,
                   const float * d_velIn_x,
                   const float * d_velIn_y,
                   const float * d_pressureIn,
                   float * d_pressureOut)
{
    solvePressure<<<blocks,threads>>>(volumeLoss, 
                                      d_levelset, 
                                      d_velIn_x, 
                                      d_velIn_y, 
                                      d_pressureIn, 
                                      d_pressureOut);
}