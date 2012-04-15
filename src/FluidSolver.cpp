#include "FluidSolver.h"
#include "FluidKernels.h"
#include <iostream>

FluidSolver::FluidSolver(int dim_x, int dim_y, int threadsPerDim, float dx) 
        : dx_(dx)
{
    dimIs(DIM_X, dim_x);
    dimIs(DIM_Y, dim_y);
    threads = dim3(threadsPerDim, threadsPerDim);
    
    //TODO make scale
    blocks = dim3(threadsPerDim, threadsPerDim);

    externalForce_ = make_float2(0.0f, -9.82f);
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
    //noSwapArrays_.resize(numVoxels);
    //surfacePoints_.resize(numVoxels);
    
    //Resize to number of blocks
    //velMag_.resize(TODO);
    
    //Build Sphere or whatever to levelset_
    //Center in the grids midpoint
    sphereCenter_ = make_float2(0.5 * dx_ * dim_[DIM_X],
                                0.5 * dx_ * dim_[DIM_Y]);
    
    //TODO, get value from gui
    sphereRadius_ = 0.25 * dx_ * dim_[DIM_X];
    
    std::cout << "FluidSolver: Building level set sphere..." << std::endl;
    buildLevelsetSphere(blocks,
                        threads,
                        sphereRadius_, 
                        sphereCenter_,
                        dx_,
                        levelset_.outPtr());
    levelset_.swap();

    std::cout << "FluidSolver: Build done..." << std::endl;
    initVolume_ = fluidVolume();
    curVolume_ = initVolume_;
    std::cout << "FluidSolver: Initialization done." << std::endl;
}

template<int T_THREADS>
void FluidSolver::solve(const float dt)
{
    float elapsed = 0.0f;
    float timestep = dt;

    //Update the fluid
    while(elapsed < dt) {
        /*unsigned char * mask = thrust::raw_pointer_cast(&mask_[0]);

        //1. Calculate the largest timestep we can take with CFL condition.
        velocityMagnitude(blocks,
                          threads,
                          thrust::raw_pointer_cast(&velMag_[0]),
                          levelset_.inPtr(),
                          vel_[DIM_X].inPtr(),
                          vel_[DIM_Y].inPtr());
        //Do thrust stuff to reduce 2nd part.
        timestep = *thrust::max_element(velMag_.begin(), velMag_.end());
         */
        //No need to solve for more than this frame
        if (timestep > (dt - elapsed))
            timestep = dt - elapsed;
        elapsed += timestep;

        /*
        //2. Extrapolate the velocity field, i.e make sure there are velocities
        //   in a band outside the fluid region.
        float2 * surfacePoints = thrust::raw_pointer_cast(&surfacePoints_[0]);
        extrapolateVelocities(blocks,
                              threads,
                              levelset_.inPtr(),
                              surfacePoints,
                              vel_[DIM_X].inPtr(),
                              vel_[DIM_Y].inPtr(),
                              vel_[DIM_X].outPtr(),
                              vel_[DIM_Y].outPtr());
        swapVelocities();
         */
        //3. Add external forces to the velocity field.
        addExternalForces(blocks,
                          threads,
                          timestep,
                          externalForce_,
                          levelset_.inPtr(),
                          vel_[DIM_X].inPtr(),
                          vel_[DIM_Y].inPtr(),
                          vel_[DIM_X].outPtr(),
                          vel_[DIM_Y].outPtr());
        swapVelocities();
        /*

        //4. Advect the velocity field in itself
        advectVelocities(blocks,
                         threads,
                         timestep,
                         levelset_.inPtr(),
                         vel_[DIM_X].inPtr(),
                         vel_[DIM_Y].inPtr(),
                         vel_[DIM_X].outPtr(),
                         vel_[DIM_Y].outPtr());
        swapVelocities();
         */
        //5. Advect the surface tracking Level Set.
        advectLevelset(blocks,
                       threads,
                       timestep,
                       1.0f / dx_,
                       noSwapArrays_.mask(),
                       levelset_.inPtr(),
                       levelset_.outPtr(),
                       vel_[DIM_X].inPtr(),
                       vel_[DIM_Y].inPtr());
        levelset_.swap();

        /*
        //6. Reinitialize the Level Set so it is numerically stable.
        reinitLevelset(blocks,
                       threads,
                       levelset_.inPtr(),
                       levelset_.outPtr(),
                       surfacePoints);
        levelset_.swap();

        //7. Calculate volume loss.
        curVolume_ = fluidVolume();

        //8. Solve pressure
        solvePressure(blocks,
                      threads,
                      initVolume_ - curVolume_,
                      levelset_.inPtr(),
                      vel_[DIM_X].inPtr(),
                      vel_[DIM_Y].inPtr(),
                      pressure_.inPtr(),
                      pressure_.outPtr());
        pressure_.swap();

        //9. Update velocity field to make it divergence free.
        updateVelocities(blocks,
                         threads,
                         levelset_.inPtr(),
                         vel_[DIM_X].inPtr(),
                         vel_[DIM_Y].inPtr(),
                         vel_[DIM_X].outPtr(),
                         vel_[DIM_Y].outPtr(),
                         pressure_.inPtr());
        swapVelocities();
        */
    }
}

void FluidSolver::solve(const float dt)
{
    switch (threads.x ) {
        case 16:
            solve<16>(dt);
            break;
        case 32:
            solve<32>(dt);
            break;
    }
}

void FluidSolver::render(uchar4 * d_pbo)
{
    writePBO(blocks, threads, d_pbo, vel_[DIM_X].inPtr());
    //raycast<<<blocks_, threads_>>>(levelset_.inPtr());
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
/*
struct volumeSum
{
    __host__ __device__
        float operator()(const float& x) const 
        {
            //If the Level Set is <= 0, then voxel is counted as water
            return x <= 0 ? 1 : 0;
        }
};*/

unsigned int FluidSolver::fluidVolume() const
{
   /*volumeSum unary_op;
    thrust::plus<float> binary_op;
    float init = 0;
    
    //Use thrust and reduce
    float temp = thrust::transform_reduce(levelset_.inVec().begin(), 
                                          levelset_.inVec().end(), 
                                          unary_op, 
                                          init, 
                                          binary_op);
    return static_cast<unsigned int>(temp);*/
}
