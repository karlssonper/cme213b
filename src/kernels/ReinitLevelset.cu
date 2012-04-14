__global__
void reinitLevelset(const float * d_levelsetIn,
                    float * d_levelsetOut,
                    float2 * d_surfacePoints)
{

}

void reinitLevelset(dim3 blocks, 
                    dim3 threads, 
                    const float * d_levelsetIn,
                    float * d_levelsetOut,
                    float2 * d_surfacePoints)
{
    reinitLevelset<<<blocks,threads>>>(d_levelsetIn,
                                       d_levelsetOut,
                                       d_surfacePoints);
}
