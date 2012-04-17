#include "TemplateMacros.h"
#include "LoadSharedMemory.h"
#include "../FluidSolver.h"

template<TEMPLATE_ARGS()>
__global__ 
void advectLevelset(const float           dt,
                    const float           inv_dx,
                    const unsigned char * d_mask,
                    const float *         d_levelsetIn,
                    float *               d_levelsetOut,
                    const float *         d_velIn_x,
                    const float *         d_velIn_y)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int g_idx = i + j * blockDim.x * gridDim.x;

    //Allocate shared memory for Level Set, +2 in for apron
    __shared__ float s_phi[(T_THREADS_X + 2) * (T_THREADS_Y + 2)];

    //Allocate memory for velocities
    __shared__ float s_vel_x[(T_THREADS_X + 1)* T_THREADS_Y];
    __shared__ float s_vel_y[(T_THREADS_Y + 1)* T_THREADS_X];
    
    //Load data
    loadValueInCenter(s_phi, d_levelsetIn, i, j);
    loadValuesAtFaces(s_vel_x, s_vel_y, d_velIn_x, d_velIn_y, i, j);

    //Sync all threads
    __syncthreads();

    int vel_idx = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    float vel_x = (s_vel_x[vel_idx] + s_vel_x[vel_idx + 1]) * 0.5f;
    float vel_y = (s_vel_y[vel_idx] + s_vel_y[vel_idx + blockDim.x + 1]) * 0.5f;
    
    float dphidx, dphidy;
    int phi_idx = centerValueCenterIdx();
    float phi = s_phi[phi_idx];
    if (vel_x > 0.0f) {
        dphidx = (phi - s_phi[centerValueLeftIdx(phi_idx)]) * inv_dx;
    } else {
        dphidx = (s_phi[centerValueRightIdx(phi_idx)] - phi) * inv_dx;
    }
    if (vel_y > 0.0f) {
        dphidy = (phi - s_phi[centerValueBelowIdx(phi_idx)]) * inv_dx;
    } else {
        dphidy = (s_phi[centerValueAboveIdx(phi_idx)] - phi) * inv_dx;
    }

    //TODO make it scale
    if (i != 0 && j != 0 && i != 255 && j != 255)
        d_levelsetOut[g_idx] = phi - dt * (dphidx * vel_x + dphidy * vel_y);
    else 
        d_levelsetOut[g_idx] = phi;
}

template<TEMPLATE_ARGS()>
void advectLevelset(FluidSolver * solver)
{
    dim3                blocks = solver->blocks();
    dim3               threads = solver->threads();
    const float             dt = solver->timestep();
    const float          invDx = 1.0f / solver->dx();
    const unsigned char * mask = solver->mask();
    const float *         lsIn = solver->levelsetIn();
    float *              lsOut = solver->levelsetOut();
    const float *           vx = solver->velocityXIn();
    const float *           vy = solver->velocityYIn();
    advectLevelset<TEMPLATE_ARGS_RUN()><<<blocks,threads>>>(dt, 
                                                            invDx, 
                                                            mask, 
                                                            lsIn, 
                                                            lsOut, 
                                                            vx, 
                                                            vy);
}

EXPLICIT_TEMPLATE_FUNCTION_INSTANTIATION(advectLevelset,FluidSolver * solver)