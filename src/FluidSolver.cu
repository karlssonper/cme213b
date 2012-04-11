#include "FluidSolver.h"

template <class T>
__device__ 
T trilerp(const T v000,
		  const T v100,
		  const T v010,
		  const T v110,
		  const T v001,
		  const T v101,
		  const T v011,
		  const T v111)
{
	return 0.0f;
}

__global__
void velocityMagnitude(float * blockMags,
					   const float * d_levelset,
		               const float * d_velIn_x,
		               const float * d_velIn_y,
		               const float * d_velIn_z)
{
	
}

__global__
void extrapolateVelocities(const float * d_levelset,
						   const float * d_velIn_x,
						   const float * d_velIn_y,
						   const float * d_velIn_z,
						   float * d_velOut_x,
						   float * d_velOut_y,
						   float * d_velOut_z)
{
	
}

__global__
void addExternalForces(const float3 force,
		               const float * d_levelset,
					   const float * d_velIn_x,
				       const float * d_velIn_y,
				       const float * d_velIn_z,
				       float * d_velOut_x,
					   float * d_velOut_y,
					   float * d_velOut_z)
{
	
}

__global__
void advectVelocities(const float dt,
					  const float * d_levelset,
					  const float * d_velIn_x,
					  const float * d_velIn_y,
					  const float * d_velIn_z,
					  float * d_velOut_x,
					  float * d_velOut_y,
					  float * d_velOut_z)
{
	
}

__global__
void advectLevelset(const float dt,
					const float * d_levelsetIn,
					float * d_levelsetOut,
					const float * d_velIn_x,
				    const float * d_velIn_y,
					const float * d_velIn_z)
{
	
}

__global__ 
void reinitLevelset(const float * d_levelsetIn,
		                   float * d_levelsetOut)
{
	
}

__global__
void solvePressure(const float volumeLoss,
				   const float * d_levelset,
				   const float * d_velIn_x,
				   const float * d_velIn_y,
				   const float * d_velIn_z,
				   const float * d_pressureIn,
				   float * d_pressureOut)
{
	
}

__global__
void updateVelocities(const float * d_levelset, 
					  const float * d_velIn_x,
			 		  const float * d_velIn_y,
					  const float * d_velIn_z,
					  float * d_velOut_x,
					  float * d_velOut_y,
					  float * d_velOut_z,
					  const float * d_pressure)
{
	
}

__global__
void buildLevelSetSphere(const float r,
						 const float3 center,
						 float * d_levelset)
{
	
}

FluidSolver::FluidSolver(uint dim_x, uint dim_y, uint dim_z)
{
	dimIs(DIM_X, dim_x);
	dimIs(DIM_Y, dim_y);
	dimIs(DIM_Z, dim_z);
	init();
}

void FluidSolver::init()
{
	const unsigned int numVoxels = dim_[DIM_X] * dim_[DIM_Y] * dim_[DIM_Z]; 
	vel_[DIM_X].resize(numVoxels);
	vel_[DIM_Y].resize(numVoxels);
	vel_[DIM_Z].resize(numVoxels);
	pressure_.resize(numVoxels);
	levelset_.resize(numVoxels);
	
	buildLevelSet();
	initVolume_ = fluidVolume();
	curVolume_ = initVolume_;
}

void FluidSolver::solve(const float dt)
{
	float elapsed = 0.0f;
	float timestep;
	
	//Update the fluid
	while(elapsed < dt) {
		//1. Calculate the largest timestep we can take with CFL condition.
		velocityMagnitude<<<blocks_, threads_>>>(	
										thrust::raw_pointer_cast(&velMag_[0]),
										levelset_.inPtr(),
										vel_[DIM_X].inPtr(),
										vel_[DIM_Y].inPtr(),
										vel_[DIM_Z].inPtr());
		//Do thrust stuff to reduce 2nd part.
		//timestep = TODO
		
		//No need to solve for more than this frame
		if (timestep > (dt - elapsed))
			timestep = dt - elapsed;
		elapsed += timestep;
		
		//2. Extrapolate the velocity field, i.e make sure there are velocities
		//   in a band outside the fluid region.
		extrapolateVelocities<<<blocks_,threads_>>>(levelset_.inPtr(),
													vel_[DIM_X].inPtr(),
													vel_[DIM_Y].inPtr(),
													vel_[DIM_Z].inPtr(),			
													vel_[DIM_X].outPtr(),
													vel_[DIM_Y].outPtr(),
													vel_[DIM_Z].outPtr());
		swapVelocities();
		
		//3. Add external forces to the velocity field.
		addExternalForces<<<blocks_, threads_>>> (externalForce_,
											      levelset_.inPtr(),
												  vel_[DIM_X].inPtr(),
											 	  vel_[DIM_Y].inPtr(),
												  vel_[DIM_Z].inPtr(),
												  vel_[DIM_X].outPtr(),
												  vel_[DIM_Y].outPtr(),
												  vel_[DIM_Z].outPtr());
		swapVelocities();
		
		//4. Advect the velocity field in itself
		advectVelocities<<<blocks_, threads_>>> (timestep,
											     levelset_.inPtr(),
												 vel_[DIM_X].inPtr(),
											 	 vel_[DIM_Y].inPtr(),
							                     vel_[DIM_Z].inPtr(),
												 vel_[DIM_X].outPtr(),
												 vel_[DIM_Y].outPtr(),
												 vel_[DIM_Z].outPtr());
		swapVelocities();
		
		//5. Advect the surface tracking Level Set.
		advectLevelset<<<blocks_,threads_>>>(timestep,
				                             levelset_.inPtr(),
				                             levelset_.outPtr(),
				                             vel_[DIM_X].inPtr(),
											 vel_[DIM_Y].inPtr(),
											 vel_[DIM_Z].inPtr());
		levelset_.swap();
		
		//6. Reinitialize the Level Set so it is numerically stable.
		reinitLevelset<<<blocks_, threads_>>>(levelset_.inPtr(),
				                              levelset_.outPtr());
		levelset_.swap();
		
		//7. Calculate volume loss.
		//Use thrust and reduce
		
		//8. Solve pressure
		solvePressure<<<blocks_,threads_>>>(initVolume_ - curVolume_,
				                            levelset_.inPtr(),
				                            vel_[DIM_X].inPtr(),
											vel_[DIM_Y].inPtr(),
											vel_[DIM_Z].inPtr(),
											pressure_.inPtr(),
											pressure_.outPtr());
		pressure_.swap();
		
		//9. Update velocity field to make it divergence free.
		updateVelocities<<<blocks_, threads_>>> (levelset_.inPtr(),
												 vel_[DIM_X].inPtr(),
												 vel_[DIM_Y].inPtr(),
												 vel_[DIM_Z].inPtr(),
												 vel_[DIM_X].outPtr(),
												 vel_[DIM_Y].outPtr(),
												 vel_[DIM_Z].outPtr(),
												 pressure_.inPtr());
		swapVelocities();
	}
}

void FluidSolver::render()
{
	
}

void FluidSolver::marchingCubes()
{
	
}

void FluidSolver::swapVelocities()
{
	for (int i = 0; i < NUM_DIMS; i++) {
		vel_[i].swap();
	}
}

void FluidSolver::dimIs(FluidSolver::Dimension d, uint value)
{
	//Is not power of two?
	if (value == 1 || (value & (value-1)) == 0) {
		std::cerr << "Dimension is not power of two." << std::endl;
		exit(1);
	}
	dim_[d] = value;
}

void FluidSolver::buildLevelSet()
{
	//Build Sphere or whatever to levelset_
}