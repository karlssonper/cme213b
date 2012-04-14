/*
 * FluidKernels.h
 *
 *  Created on: Apr 14, 2012
 *      Author: per
 */

#ifndef FLUIDKERNELS_H_
#define FLUIDKERNELS_H_

#include "kernels/AdvectLevelset.cu"
#include "kernels/VelocityMagnitude.cu"
#include "kernels/ExtrapolateVelocities.cu"
#include "kernels/AddExternalForces.cu"
#include "kernels/AdvectVelocities.cu"
extern "C" advectLevelset();

extern "C" void velocityMagnitude(dim3 blocks, dim3 threads, float * blockMags,
                                  const float * d_levelset,
                                  const float * d_velIn_x,
                                  const float * d_velIn_y);

extern "C" void extrapolateVelocities(dim3 blocks, dim3 threads,
                                      const float * d_levelset,
                                      const float2 * d_surfacePoints,
                                      const float * d_velIn_x,
                                      const float * d_velIn_y,
                                      float * d_velOut_x,
                                      float * d_velOut_y);


extern "C" void addExternalForces(dim3 blocks, dim3 threads, const float dt,
                                  const float2 force,
                                  const float * d_levelset,
                                  const float * d_velIn_x,
                                  const float * d_velIn_y,
                                  float * d_velOut_x,
                                  float * d_velOut_y);

extern "C" void advectVelocities(dim3 blocks, dim3 threads,const float dt,
                                 const float * d_levelset,
                                 const float * d_velIn_x,
                                 const float * d_velIn_y,
                                 float * d_velOut_x,
                                 float * d_velOut_y)


dim3 blocks;
dim3 threads;

#endif /* FLUIDKERNELS_H_ */
