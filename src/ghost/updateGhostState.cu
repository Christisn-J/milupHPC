#if PERIODIC_BOUNDARIES
#include "../../include/ghost/updateGhostState.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void updateGhosts(Particles *particles, IntegratedParticles *ghosts){
/// TODO: optimization: integrate a flag to update only what is necessary
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            for (integer offset = 0; (bodyIndex + offset)< *(particles->numParticles)*MAX_GHOSTS_PER_PARTICLE; offset += stride){
                if (particles->mapGhost[(bodyIndex + offset)] >= 0){
                    ghosts->massGhosts[particles->mapGhost[(bodyIndex + offset)]] = particles->mass[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->rho[particles->mapGhost[(bodyIndex + offset)]] = particles->rho[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->p[particles->mapGhost[(bodyIndex + offset)]] = particles->p[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->e[particles->mapGhost[(bodyIndex + offset)]] = particles->e[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->vx[particles->mapGhost[(bodyIndex + offset)]] = particles->vx[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
#if DIM > 1
                    ghosts->vy[particles->mapGhost[(bodyIndex + offset)]] = particles->vy[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
#if DIM == 3
                    ghosts->vz[particles->mapGhost[(bodyIndex + offset)]] = particles->vz[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
#endif // 3D
#endif // 2D    
                }
            }
        }

        real Launch::updateGhosts(Particles *particles,IntegratedParticles *ghosts){
            ExecutionPolicy executionPolicy(1,1);
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::updateGhosts, particles, ghosts);
        }
    }
}
#endif