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
#include "DeviceArraysNoSwap.h"

class FluidSolver
{
public:
    FluidSolver(int dim_x, int dim_y, int threadsPerDim, float dx);
    void init();
    void solve (const float dt);
    void render(uchar4 * d_pbo);
    void marchingCubes();
protected:
    enum Dimension{ DIM_X = 0, DIM_Y = 1, NUM_DIMS = 2 };
    unsigned int dim_[NUM_DIMS];
    float dx_;
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

    template<int T_THREADS>
    void solve(const float dt);
private:
    FluidSolver();
    FluidSolver(const FluidSolver &);
    void operator=(const FluidSolver &);
};

#endif /* FLUIDSOLVER_H_ */
