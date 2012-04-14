/*
 * FluidKernels.h
 *
 *  Created on: Apr 14, 2012
 *      Author: per
 */

#ifndef FLUIDKERNELS_H_
#define FLUIDKERNELS_H_

#include "kernels/advectLevelset.cu"

extern "C" advectLevelset();

dim3 blocks;
dim3 threads;

#endif /* FLUIDKERNELS_H_ */
