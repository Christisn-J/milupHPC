#include "../../include/ghost/addGhostDensity.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void addGhostDensity(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticleHandler *ghosts){
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            
            real W, Wj, dx[DIM], dWdx[DIM], dWdr;
            real distance;
            int ip;

            // iterate over all particles
            for(integer offset = 0; (bodyIndex + offset) < particles->numParticles; offset += stride){
                // iterate over all possible ghost particles
                for (int iGhost =0; iGhost< particles->noiGhost[(bodyIndex + offset)]; ++iGhost) {
                    ip = particles->nnlGhost[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + iGhost];

                        // calculate dimensionless the difference
                        dx = particles->x[bodyIndex + offset] - ghosts->x[ip];
#if DIM > 1
                        dy = particles->y[bodyIndex + offset] - ghosts->y[ip];
#if DIM == 3
                        dz = particles->z[bodyIndex + offset] - ghosts->z[ip];
#endif // 3D
#endif // 2D
                        // calculate the distance depending on the dimension
#if DIM == 1
                        distance = dx * dx;
#elif DIM == 2
                        distance = dx * dx + dy * dy;
#else //DIM == 3
                        distance = dx * dx + dy * dy + dz * dz;
#endif


/// TODO: use kernal
                    // Attention: initilase gohst particles (m) and after initilase particles
                    real W, Wj, dx[DIM], dWdx[DIM], dWdr;
                    particles->rho[(bodyIndex + offset)] += ghosts->m[ip]*kernel(distance,particles->sml);
                }
            if (numInteractions > MAX_NUM_INTERACTIONS) {
                // Issue massage
                cudaTerminate("Particle %i has zero or negative density  ", numInteractions, MAX_NUM_INTERACTIONS);
            }
            }
        }

        real Launch::addGhostDensity(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticleHandler *ghosts) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::addGhostDensity, kernel, particles, ghosts);
        }
    }
}