#include "../FluidSolver.h"

__global__
void buildLevelsetSphere(const float r,
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

void buildLevelsetSphere(FluidSolver * solver)
{
    dim3                blocks = solver->blocks();
    dim3               threads = solver->threads();
    const float             dx = solver->dx();
    const float              r = solver->sphereRadius();
    const float2        center = solver->sphereCenter();
    float *              lsOut = solver->levelsetOut();
    buildLevelsetSphere<<<blocks,threads>>>(r, center, dx, lsOut);
}