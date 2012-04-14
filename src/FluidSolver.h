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

#include "DeviceArray.h"

class FluidSolver
{
public:
    FluidSolver(int dim_x, int dim_y, int threadsPerDim, float dx);
    void init();
    void solve (const float dt);
    void render();
    void marchingCubes();
protected:
    enum Dimension{ DIM_X = 0, DIM_Y = 1, NUM_DIMS = 2 };
    unsigned int dim_[NUM_DIMS];
    float dx_;
    DeviceArray<float> vel_[NUM_DIMS];
    DeviceArray<float> pressure_;
    DeviceArray<float> levelset_;

    thrust::device_vector<unsigned char> mask_;
    thrust::device_vector<float> velMag_;
    thrust::device_vector<float2> surfacePoints_;
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
