#include "TemplateMacros.h"
#include "LoadSharedMemory.h"
#include "UtilDeviceFunctions.h"
#include "../FluidSolver.h"
#include <cfloat>

template<Direction dir>
__device__
void minDist(const float phi,
             const float * s_phi,
             const int center_idx,
             const float dx,
             const float2 & world_pos,
             float & min_dist, 
             short & min_dir)
{
    float phiNeighbor;
    if (dir == DIR_LEFT) {
        phiNeighbor = s_phi[centerValueLeftIdx(center_idx)];
    } else if (dir == DIR_RIGHT) {
        phiNeighbor = s_phi[centerValueRightIdx(center_idx)];
    } else if (dir == DIR_UP) {
        phiNeighbor = s_phi[centerValueAboveIdx(center_idx)];
    } else if (dir == DIR_DOWN) {
        phiNeighbor = s_phi[centerValueBelowIdx(center_idx)];
    }
    
    if (phi * phiNeighbor <= 0) {
        float dist;
        if (dir == DIR_LEFT) {
            dist = world_pos.x - (1-(abs(phi)/(abs(phi)+abs(phiNeighbor))))*dx;
        } else if (dir == DIR_RIGHT) {
            dist = world_pos.x + (1-(abs(phi)/(abs(phi)+abs(phiNeighbor))))*dx;
        } else if (dir == DIR_UP) {
            dist = world_pos.y + (1-(abs(phi)/(abs(phi)+abs(phiNeighbor))))*dx;
        } else if (dir == DIR_DOWN) {
            dist = world_pos.y - (1-(abs(phi)/(abs(phi)+abs(phiNeighbor))))*dx;
        }
        dist *=dist;
        if (dist < min_dist) {
            min_dist = dist;
            min_dir = dir;
        }
    }
    
}

template<TEMPLATE_ARGS()>
__global__ 
void findSurface(const float dx,
                 const float * d_levelsetIn,
                 float * d_levelsetOut,
                 float2 * d_surfacePointsOut)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    //Allocate shared memory for Level Set, +2 in for apron
    __shared__ float s_phi[(T_THREADS_X + 2) * (T_THREADS_Y + 2)];
    
    //Load data
    loadValueInCenter(s_phi, d_levelsetIn, i, j);
    
    //Sync all threads
    __syncthreads();
    
    int center_idx = centerValueCenterIdx();
    
    //Set min_dist to something big;
    float min_dist = FLT_MAX;
    short min_dir = DIR_UNDEFINED;
    
    //If the product between two neighboring phi values is <= 0 then surface!
    float phi = s_phi[center_idx];
    
    //World Position
    float2 wp = indexToWorld(i, j, dx);
    
    minDist<DIR_LEFT>(phi, s_phi, center_idx, dx, wp, min_dist, min_dir);
    minDist<DIR_RIGHT>(phi, s_phi,center_idx, dx, wp, min_dist, min_dir);
    minDist<DIR_UP>(phi, s_phi,center_idx, dx, wp, min_dist, min_dir);
    minDist<DIR_DOWN>(phi, s_phi,center_idx, dx, wp, min_dist, min_dir);

    const int g_idx = i + j * blockDim.x * gridDim.x;
    if (min_dist != FLT_MAX) {
        switch(min_dir) {
            case DIR_LEFT:
                d_surfacePointsOut[g_idx] = make_float2(wp.x - sqrt(min_dist), 
                                                        wp.y);
                break;
            case DIR_RIGHT:
                d_surfacePointsOut[g_idx] = make_float2(wp.x + sqrt(min_dist), 
                                                        wp.y);
                break;
            case DIR_UP:
                d_surfacePointsOut[g_idx] = make_float2(wp.x, 
                                                        wp.y + sqrt(min_dist));
                break;
            case DIR_DOWN:
                d_surfacePointsOut[g_idx] = make_float2(wp.x, 
                                                        wp.y - sqrt(min_dist));
                break;
        }
    } else  {
        //Since we don't have any negative coordinates in the grid
        d_surfacePointsOut[g_idx] = make_float2(-1.0f, -1.0f);
    }
}



__global__
void reinitLevelset(const float * d_levelsetIn,
                    float * d_levelsetOut,
                    float2 * d_surfacePoints)
{
    
}

template<TEMPLATE_ARGS()>
void reinitLevelset(FluidSolver * solver)
{
    dim3                blocks = solver->blocks();
    dim3               threads = solver->threads();
    
    //1. Find surface
    
    //findSurfacePoints<T_THREADS_X, T_THREADS_Y><<<blocks,threads>>>()
    
    //2. Propogate levelset and surfacepoints
    //3. Final bla
    
}

EXPLICIT_TEMPLATE_FUNCTION_INSTANTIATION(reinitLevelset,FluidSolver * solver)