#include "../FluidSolver.h"

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
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int index = i + j * blockDim.x * gridDim.x;

    // Only compute external forces for fluid voxels
 //   if (d_levelset(index))
    {
        d_velOut_x[index] = d_velIn_x[index] + dt*force.x;
        d_velOut_y[index] = d_velIn_y[index] + dt*force.y;
    }
}

void addExternalForces(FluidSolver * solver)
{
    dim3                blocks = solver->blocks();
    dim3               threads = solver->threads();
    const float             dt = solver->timestep();
    const float2          force = solver->externalForce();
    const float *         lsIn = solver->levelsetIn();
    const float *         vxIn = solver->velocityXIn();
    const float *         vyIn = solver->velocityYIn();
    float *              vxOut = solver->velocityXOut();
    float *              vyOut = solver->velocityYOut();
    addExternalForces<<<blocks, threads>>>(dt,force,lsIn,vxIn,vyIn,vxOut,vyOut);
}
