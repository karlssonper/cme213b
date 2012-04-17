/*
 * FluidSolver.h
 *
 *  Created on: Apr 10, 2012
 *      Author: per
 */

#ifndef FLUIDSOLVER_H_
#define FLUIDSOLVER_H_

#define MASK_SOLID 0
#define MASK_FLUID 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "DeviceArray.h"

class FluidSolver
{
public:
    FluidSolver(int dim_x, int dim_y, int threadsPerDim, float dxWorld);
    void init();
    void solve (const float dt);
    void render(uchar4 * d_pbo);
    void marchingCubes();

    //Inlined accessors
    float timestep() const { return curTimestep_; };
    float dx() const { return dx_; };
    dim3 blocks() const { return blocks_; };
    dim3 threads() const { return threads_; };
    float2 externalForce() const { return externalForce_; };
    const float * levelsetIn() const { return levelset_.inPtr(); };
    float * levelsetOut() { return levelset_.outPtr(); };
    const float * velocityXIn() const { return vel_[DIM_X].inPtr(); };
    const float * velocityYIn() const { return vel_[DIM_Y].inPtr(); };
    float * velocityXOut() { return vel_[DIM_X].outPtr(); };
    float * velocityYOut() { return vel_[DIM_Y].outPtr(); };
    unsigned char * mask() { return noSwapArrays_.mask(); };
    const float2 * surfacePointsIn() const;
    float2 * surfacePointsOut();
    float  sphereRadius() const { return sphereRadius_; };
    float2  sphereCenter() const { return sphereCenter_; };
    const float * pressureIn() const { return pressure_.inPtr(); };
    float * pressureOut() { return pressure_.outPtr(); };
    float * velocityMag() { return noSwapArrays_.velocityMag(); };
protected:
    enum Dimension{ DIM_X = 0, DIM_Y = 1, NUM_DIMS = 2 };
    unsigned int dim_[NUM_DIMS];
    dim3 blocks_;
    dim3 threads_;
    float dx_;
    float curTimestep_;
    DeviceArray vel_[NUM_DIMS];
    DeviceArray pressure_;
    DeviceArray levelset_;
    DeviceArraysNoSwap noSwapArrays_;
    unsigned int initVolume_;
    unsigned int curVolume_;
    float2 externalForce_;
    float sphereRadius_;
    float2 sphereCenter_;
    void dimIs(Dimension d, unsigned int value);
    void swapVelocities();
    void buildLevelSet();
    unsigned int fluidVolume() const;


    template<int T_THREADS_X>
    void solve(const float dt);
    template<int T_THREADS_X, int T_THREADS_Y>
    void solve(const float dt);
private:
    FluidSolver();
    FluidSolver(const FluidSolver &);
    void operator=(const FluidSolver &);
};

#endif /* FLUIDSOLVER_H_ */
