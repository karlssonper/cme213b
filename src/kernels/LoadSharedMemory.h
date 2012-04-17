/*
 * LoadSharedMemory.h
 *
 *  Created on: Apr 16, 2012
 *      Author: per
 */

#ifndef LOADSHAREDMEMORY_H_
#define LOADSHAREDMEMORY_H_

__forceinline__ __device__
int centerValueCenterIdx()
{
    return threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2);
}


__forceinline__ __device__
int centerValueLeftIdx(int centerIdx)
{
    return --centerIdx;
}

__forceinline__ __device__
int centerValueRightIdx(int centerIdx)
{
    return ++centerIdx;
}

__forceinline__ __device__
int centerValueAboveIdx(int centerIdx)
{
    return centerIdx + (blockDim.x + 2);
}

__forceinline__ __device__
int centerValueBelowIdx(int centerIdx)
{
    return centerIdx + (blockDim.x + 2);
}


template<typename T>
__device__
void loadValueInCenter(T * s_out, const T * g_in, int i, int j)
{
    const int g_idx = i + j * blockDim.x * gridDim.x;

    //Load inner phi
    int s_idx = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2);
    s_out[s_idx] = g_in[g_idx];

    //Load phi at the apron
    //Left boundary
    if (threadIdx.x == 0 && blockIdx.x != 0) {
        s_idx = (threadIdx.y + 1) * (blockDim.x + 2);
        s_out[s_idx] = g_in[g_idx - 1];
    }
    //Right boundary
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1) {
        s_idx = (threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2;
        s_out[s_idx] = g_in[g_idx + 1];
    }
    //Bottom boundary
    if (threadIdx.y == 0 && blockIdx.y != 0) {
        s_idx = threadIdx.x + 1;
        s_out[s_idx] = g_in[g_idx - gridDim.x * blockDim.x];
    }
    //Top boundary
    if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1) {
        s_idx = (threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1;
        s_out[s_idx] = g_in[g_idx + gridDim.x * blockDim.x];
    }
}

template<typename T>
__device__
void loadValuesAtFaces(T * s_out_x,
                       T * s_out_y,
                       const T * g_in_x,
                       const T * g_in_y,
                       int i,
                       int j)
{
    int s_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    //Because of MaC grid, global memeory has one extra component
    int g_idx_vel = i + j * (blockDim.x * gridDim.x + 1);

    //Load inner velocities
    s_out_x[s_idx] = g_in_x[g_idx_vel];
    s_out_y[s_idx] = g_in_y[g_idx_vel];

    //Load boundary velocities
    //Right boundary
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1) {
        s_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x + 1;
        s_out_x[s_idx] = g_in_x[g_idx_vel + 1];
    }
    //Top boundary
    if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1) {
        s_idx = (threadIdx.y + 1) * (blockDim.x + 1) + threadIdx.x;
        s_out_y[s_idx] = g_in_y[g_idx_vel + blockDim.x * gridDim.x + 1];
    }
}

#endif /* LOADSHAREDMEMORY_H_ */
