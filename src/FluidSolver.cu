#include "FluidSolver.h"

template <class T>
__device__ 
T bilerp(const T v00,
		 const T v10,
		 const T v01,
		 const T v11)
{
	return 0.0f;
}

__device__
unsigned int worldToIdx2D()
{
	
}

__device__
float3 idx2DToWorld()
{
	
}

__global__
void velocityMagnitude(float * blockMags,
					   const float * d_levelset,
		               const float * d_velIn_x,
		               const float * d_velIn_y)
{
	
}

__global__
void extrapolateVelocities(const float * d_levelset,
		                   const float2 * d_surfacePoints,
						   const float * d_velIn_x,
						   const float * d_velIn_y,
						   float * d_velOut_x,
						   float * d_velOut_y)
{
	
}

__global__
void addExternalForces(const float dt,
		               const float2 force,
		               const float * d_levelset,
					   const float * d_velIn_x,
				       const float * d_velIn_y,
				       float * d_velOut_x,
					   float * d_velOut_y)
{

    // Get Index
    // Notes on indexing:
    int index  = blockDim.x* (blockIdx.x + blockIdx.y*gridDim.x) +
            threadIdx.y*blockDim.x + threadIdx.x;

    // Only compute external forces for fluid voxels
 //   if (d_levelset(index))
    {
        d_velOut_x[index] = d_velIn_x[index] + dt*force.x;
        d_velOut_y[index] = d_velIn_y[index] + dt*force.y;
    }
	
}

__global__
void advectVelocities(const float dt,
					  const float * d_levelset,
					  const float * d_velIn_x,
					  const float * d_velIn_y,
					  float * d_velOut_x,
					  float * d_velOut_y)
{
	
}

__global__
void advectLevelset(const float dt,
					const float * d_levelsetIn,
					float * d_levelsetOut,
					const float * d_velIn_x,
				    const float * d_velIn_y)
{
	
}

__global__ 
void reinitLevelset(const float * d_levelsetIn,
		            float * d_levelsetOut,
		            float2 * surfacePoints)
{
	
}

__global__
void solvePressure(const float volumeLoss,
				   const float * d_levelset,
				   const float * d_velIn_x,
				   const float * d_velIn_y,
				   const float * d_pressureIn,
				   float * d_pressureOut)
{
	
}

__global__
void updateVelocities(const float * d_levelset, 
					  const float * d_velIn_x,
			 		  const float * d_velIn_y,
					  float * d_velOut_x,
					  float * d_velOut_y,
					  const float * d_pressure)
{
	
}

__global__
void buildLevelSetSphere(const float r,
						 const float2 center,
						 float * d_levelset)
{
	//Get idx;
	int i,j,k;
	float phi;
	
	
	//d_levelset[idx1D] = phi;
}

__global__
void raycast(const float * d_levelset)
{
	
}

FluidSolver::FluidSolver(uint dim_x, uint dim_y)
{
	dimIs(DIM_X, dim_x);
	dimIs(DIM_Y, dim_y);
	init();
}

void FluidSolver::init()
{
	std::cout << "FluidSolver: Initiating a new domain, " <<  
	   dim_[DIM_X] << " x " << dim_[DIM_Y] << std::endl;
	const unsigned int numVoxels = dim_[DIM_X] * dim_[DIM_Y]; 
	vel_[DIM_X].resize(numVoxels);
	vel_[DIM_Y].resize(numVoxels);
	pressure_.resize(numVoxels);
	levelset_.resize(numVoxels);
	surfacePoints_.resize(numVoxels);
	
	//Resize to number of blocks
	//velMag_.resize(TODO);
	
	//Build Sphere or whatever to levelset_
	//Center in the grids midpoint
	sphereCenter_ = make_float2(0.5 * dx_ * dim_[DIM_X],
								0.5 * dx_ * dim_[DIM_Y]);
	
	//TODO, get value from gui
	sphereRadius_ = 0.25 * dx_ * dim_[DIM_X];
	
	std::cout << "FluidSolver: Building level set sphere..." << std::endl;
	buildLevelSetSphere<<<blocks_, threads_>>>(sphereRadius_, 
											   sphereCenter_,
											   levelset_.outPtr());
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
										vel_[DIM_Y].inPtr());
		//Do thrust stuff to reduce 2nd part.
		timestep = *thrust::max_element(velMag_.begin(), velMag_.end());
		
		//No need to solve for more than this frame
		if (timestep > (dt - elapsed))
			timestep = dt - elapsed;
		elapsed += timestep;
		
		//2. Extrapolate the velocity field, i.e make sure there are velocities
		//   in a band outside the fluid region.
		float2 * surfacePoints = thrust::raw_pointer_cast(&surfacePoints_[0]);
		extrapolateVelocities<<<blocks_,threads_>>>(levelset_.inPtr(),
													surfacePoints,
													vel_[DIM_X].inPtr(),
													vel_[DIM_Y].inPtr(),			
													vel_[DIM_X].outPtr(),
													vel_[DIM_Y].outPtr());
		swapVelocities();
		
		//3. Add external forces to the velocity field.
		addExternalForces<<<blocks_, threads_>>> (timestep,
												  externalForce_,
											      levelset_.inPtr(),
												  vel_[DIM_X].inPtr(),
											 	  vel_[DIM_Y].inPtr(),
												  vel_[DIM_X].outPtr(),
												  vel_[DIM_Y].outPtr());
		swapVelocities();
		
		//4. Advect the velocity field in itself
		advectVelocities<<<blocks_, threads_>>> (timestep,
											     levelset_.inPtr(),
												 vel_[DIM_X].inPtr(),
											 	 vel_[DIM_Y].inPtr(),
												 vel_[DIM_X].outPtr(),
												 vel_[DIM_Y].outPtr());
		swapVelocities();
		
		//5. Advect the surface tracking Level Set.
		advectLevelset<<<blocks_,threads_>>>(timestep,
				                             levelset_.inPtr(),
				                             levelset_.outPtr(),
				                             vel_[DIM_X].inPtr(),
											 vel_[DIM_Y].inPtr());
		levelset_.swap();
		
		//6. Reinitialize the Level Set so it is numerically stable.
		reinitLevelset<<<blocks_, threads_>>>(levelset_.inPtr(),
				                              levelset_.outPtr(),
				                              surfacePoints);
		levelset_.swap();
		
		//7. Calculate volume loss.
		curVolume_ = fluidVolume();
		
		//8. Solve pressure
		solvePressure<<<blocks_,threads_>>>(initVolume_ - curVolume_,
				                            levelset_.inPtr(),
				                            vel_[DIM_X].inPtr(),
											vel_[DIM_Y].inPtr(),
											pressure_.inPtr(),
											pressure_.outPtr());
		pressure_.swap();
		
		//9. Update velocity field to make it divergence free.
		updateVelocities<<<blocks_, threads_>>> (levelset_.inPtr(),
												 vel_[DIM_X].inPtr(),
												 vel_[DIM_Y].inPtr(),
												 vel_[DIM_X].outPtr(),
												 vel_[DIM_Y].outPtr(),
												 pressure_.inPtr());
		swapVelocities();
	}
}

void FluidSolver::render()
{
	raycast<<<blocks_, threads_>>>(levelset_.inPtr());
}

void FluidSolver::marchingCubes()
{
	//Add in the future. Needed to make a triangular mesh from implicit surface
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
	/*TODO
	 * if (value == 1 || (value & (value-1)) == 0) {
		std::cerr << "Dimension " << value 
				<< " is not power of two." << std::endl;
		exit(1);
	}*/
	dim_[d] = value;
}

struct volumeSum
{
    __host__ __device__
        float operator()(const float& x) const 
        {
			//If the Level Set is <= 0, then voxel is counted as water
            return x <= 0 ? 1 : 0;
        }
};

unsigned int FluidSolver::fluidVolume() const
{
	volumeSum unary_op;
	thrust::plus<float> binary_op;
	float init = 0;
	
	//Use thrust and reduce
	float temp = thrust::transform_reduce(levelset_.inVec().begin(), 
										  levelset_.inVec().end(), 
										  unary_op, 
										  init, 
										  binary_op);
	return static_cast<unsigned int>(temp);
}
