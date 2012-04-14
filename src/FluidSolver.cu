#include "FluidSolver.h"

template <class T>
__device__
T bilerp(const T v00,
         const T v10,
         const T v01,
         const T v11)
{
    return v00;
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

template<int T_THREADS>
__global__
void advectLevelset(const float dt,
                    const float inv_dx,
                    const unsigned char * d_mask,
                    const float * d_levelsetIn,
                    float * d_levelsetOut,
                    const float * d_velIn_x,
                    const float * d_velIn_y)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int g_idx = i + j * blockDim.x * gridDim.x;

    //Allocate shared memory for Level Set, +2 in for apron
    __shared__ float s_phi[(T_THREADS + 2) * (T_THREADS + 2)];

    //Load inner phi
    int s_idx = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2);
    s_phi[s_idx] = d_levelsetIn[g_idx];

    //Load phi at the apron
    //Left boundary
    if (threadIdx.x == 0 && blockIdx.x != 0) {
        s_idx = (threadIdx.y + 1) * (blockDim.x + 2);
        s_phi[s_idx] = d_levelsetIn[g_idx - 1];
    }
    //Right boundary
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1) {
        s_idx = (threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2;
        s_phi[s_idx] = d_levelsetIn[g_idx + 1];
    }
    //Bottom boundary
    if (threadIdx.y == 0 && blockIdx.y != 0) {
        s_idx = threadIdx.x + 1;
        s_phi[s_idx] = d_levelsetIn[g_idx - gridDim.x * blockDim.x];
    }
    //Top boundary
    if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1) {
        s_idx = (threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1;
        s_phi[s_idx] = d_levelsetIn[g_idx + gridDim.x * blockDim.x];
    }
    //Sync all threads
    __syncthreads();

    //Allocate memory for velocities
    __shared__ float s_vel_x[(T_THREADS + 1)*(T_THREADS + 1)];
    __shared__ float s_vel_y[(T_THREADS + 1)*(T_THREADS + 1)];

    s_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    //Because of MaC grid, global memeory has one extra component
    int g_idx_vel = i * j * (blockDim.x * gridDim.x + 1);

    //Load inner velocities
    s_vel_x[s_idx] = d_velIn_x[g_idx_vel];
    s_vel_y[s_idx] = d_velIn_y[g_idx_vel];

    //Load boundary velocities
    //Right boundary
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1) {
        s_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x + 1;
        s_vel_x[s_idx] = d_velIn_x[g_idx_vel + 1];
    }
    //Top boundary
    if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1) {
        s_idx = (threadIdx.y + 1) * (blockDim.x + 1) + threadIdx.x;
        s_vel_x[s_idx] = d_velIn_x[g_idx_vel + blockDim.x * gridDim.x + 1];
    }

    //Sync all threads
    __syncthreads();

    int vel_idx = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    float vel_x = (s_vel_x[vel_idx] + s_vel_x[vel_idx + 1]) * 0.5f;
    float vel_y = (s_vel_y[vel_idx] + s_vel_y[vel_idx + blockDim.x + 1]) * 0.5f;

    float dphidx, dphidy;
    int phi_idx = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2);
    float phi = s_phi[phi_idx];
    if (vel_x > 0.0f) {
        dphidx = (phi - s_phi[phi_idx - 1]) * inv_dx;
    } else {
        dphidx = (s_phi[phi_idx + 1] - phi) * inv_dx;
    }
    if (vel_y > 0.0f) {
        dphidy = (phi - s_phi[phi_idx - (blockDim.x + 2)]) * inv_dx;
    } else {
        dphidy = (s_phi[phi_idx + (blockDim.x + 2)] - phi) * inv_dx;
    }

    d_levelsetOut[g_idx] = phi - dt * (dphidx * vel_x + dphidy * vel_y);
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
                         const float dx,
                         float * d_levelset)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = i + j * blockDim.x * gridDim.x;

    const float x = dx * i - center.x;
    const float y = dx * j - center.y;
    d_levelset[idx] = sqrt(x*x+y*y) - r;
}

__global__
void raycast(const float * d_levelset)
{

}

FluidSolver::FluidSolver(int dim_x, int dim_y, int threadsPerDim, float dx) 
        : dx_(dx)
{
    dimIs(DIM_X, dim_x);
    dimIs(DIM_Y, dim_y);
    threads_ = dim3(threadsPerDim, threadsPerDim); 
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
                                               dx_,
                                               levelset_.outPtr());
    initVolume_ = fluidVolume();
    curVolume_ = initVolume_;
}

template<int T_THREADS>
void FluidSolver::solve(const float dt)
{
    float elapsed = 0.0f;
    float timestep;

    //Update the fluid
    while(elapsed < dt) {
        unsigned char * mask = thrust::raw_pointer_cast(&mask_[0]);

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
        advectLevelset<T_THREADS> <<<blocks_,threads_>>>(timestep,
                                             1.0f / dx_,
                                             mask,
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

void FluidSolver::solve(const float dt)
{
    switch (threads_.x ) {
        case 16:
            solve<16>(dt);
            break;
        case 32:
            solve<32>(dt);
            break;
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
