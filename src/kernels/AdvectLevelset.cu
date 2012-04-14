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

void advectLevelset()
{
    
}