#include "../../include/ghost/updateGhostState.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void updateGhostState(Particles *particles, IntegratedParticleHandler *ghosts){
/// TODO: optimization: integrate a flag to update only what is necessary
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            for (integer offset = 0; (bodyIndex + offset)<particles->numParticles*MAX_GHOSTS_PER_PARTICLE; offset += stride){
                if (particles->ghostMap[(bodyIndex + offset)] >= 0){
                    ghosts->m[particles->ghostMap[(bodyIndex + offset)]] = particles->m[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->rho[particles->ghostMap[(bodyIndex + offset)]] = particles->rho[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->p[particles->ghostMap[(bodyIndex + offset)]] = particles->p[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->e[particles->ghostMap[(bodyIndex + offset)]] = particles->e[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
                    ghosts->vx[particles->ghostMap[(bodyIndex + offset)]] = particles->vx[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
#if DIM > 1
                    ghosts->vy[particles->ghostMap[(bodyIndex + offset)]] = particles->vy[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
#if DIM == 3
                    ghosts->vz[particles->ghostMap[(bodyIndex + offset)]] = particles->vz[(bodyIndex + offset)/MAX_GHOSTS_PER_PARTICLE];
#endif // 3D
#endif // 2D    
                }
            }
        }

        real Launch::updateGhostState(Particles *particles,IntegratedParticleHandler *ghosts){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::updateGhostState, particles, ghosts);
        }
    }
}