/*
 * FluidKernels.h
 *
 *  Created on: Apr 14, 2012
 *      Author: per
 */

#ifndef FLUIDKERNELS_H_
#define FLUIDKERNELS_H_

dim3 blocks;
dim3 threads;

void addExternalForces(FluidSolver * solver);

void advectVelocities(dim3 blocks,
                      dim3 threads,
                      const float dt,
                      const float * d_levelset,
                      const float * d_velIn_x,
                      const float * d_velIn_y,
                      float * d_velOut_x,
                      float * d_velOut_y);

template<int T_THREADS_X, int T_THREADS_Y>
void advectLevelset(FluidSolver * solver);

void buildLevelsetSphere(FluidSolver * solver);

void extrapolateVelocities(dim3 blocks,
                           dim3 threads,
                           const float * d_levelset,
                           const float2 * d_surfacePoints,
                           const float * d_velIn_x,
                           const float * d_velIn_y,
                           float * d_velOut_x,
                           float * d_velOut_y);

template<int T_THREADS_X, int T_THREADS_Y>
void reinitLevelset(FluidSolver * solver);

void solvePressure(dim3 blocks,
                   dim3 threads,
                   const float volumeLoss,
                   const float * d_levelset,
                   const float * d_velIn_x,
                   const float * d_velIn_y,
                   const float * d_pressureIn,
                   float * d_pressureOut);

void updateVelocities(dim3 blocks,
                      dim3 threads,
                      const float * d_levelset,
                      const float * d_velIn_x,
                      const float * d_velIn_y,
                      float * d_velOut_x,
                      float * d_velOut_y,
                      const float * d_pressure);

void velocityMagnitude(dim3 blocks,
                       dim3 threads,
                       float * blockMags,
                       const float * d_levelset,
                       const float * d_velIn_x,
                       const float * d_velIn_y);

void writePBO(dim3 blocks,
              dim3 threads,
              uchar4 * d_pbo,
              const float * d_levelset);


#endif /* FLUIDKERNELS_H_ */
