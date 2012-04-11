/*
 * FluidSolver.h
 *
 *  Created on: Apr 10, 2012
 *      Author: per
 */

#ifndef FLUIDSOLVER_H_
#define FLUIDSOLVER_H_

#include "DeviceArray.h"

class FluidSolver
{
public:
	FluidSolver(unsigned int dim_x, unsigned int dim_y, unsigned int dim_z);
	void init();
	void solve (const float dt);
	void render();
	void marchingCubes();
protected:
	enum Dimension{ DIM_X = 0, DIM_Y = 1, DIM_Z = 2, NUM_DIMS = 3 };
	unsigned int dim_[NUM_DIMS];
	DeviceArray<float> vel_[NUM_DIMS];
	DeviceArray<float> pressure_;
	DeviceArray<float> levelset_;
	unsigned int initVolume_;
	unsigned int curVolume_;
	float3 externalForce_;
	dim3 blocks_;
	dim3 threads_;
	void dimIs(Dimension d, unsigned int value);
	void swapVelocities();
	void buildLevelSet();
	unsigned int fluidVolume() const;
private:
	FluidSolver();
	FluidSolver(const FluidSolver &);
	void operator=(const FluidSolver &);
};

#endif /* FLUIDSOLVER_H_ */
