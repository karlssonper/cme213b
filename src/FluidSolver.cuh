/*
 * FluidSolvers.h
 *
 *  Created on: Apr 13, 2012
 *      Author: per
 */

#ifndef FLUIDSOLVER_CUH_
#define FLUIDSOLVER_CUH_

namespace FluidSolver
{
	velocityMagnitude(float * blockMags,
					  const float * d_levelset,
		              const float * d_velIn_x,
		              const float * d_velIn_y);
	
	
}

#endif /* FLUIDSOLVER_CUH_ */
