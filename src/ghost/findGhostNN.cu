#include "../../include/ghost/findGhostNN.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void fixedRadiusNNGhost_bruteForce(Particles *particles, Particles *ghosts){
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            real dx, dy, dz;
            real distance;
            int numInteractions;

            // iterate over all particles
            for(integer offset = 0; (bodyIndex + offset)<particles->numParticles; offset += stride){

                numInteractions = 0;
                // iterate over all possible ghost particles
/// TODO: optimization search: ghost into tree or generate own tree for ghosts
                for (int iGhost=0; i<ghosts->numParticles; ++iGhost) {
                    // no self interaction, because |r_o-r_i| would be zero
                    if ((bodyIndex + offset) != i) {

                        // calculate dimensionless the difference
                        dx = particles->x[bodyIndex + offset] - ghosts->x[iGhost];
#if DIM > 1
                        dy = particles->y[bodyIndex + offset] - ghosts->y[iGhost];
#if DIM == 3
                        dz = particles->z[bodyIndex + offset] - ghosts->z[iGhost];
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

                        // condition distance < h² for both particle, necessary because of dynamic smoothing length
                        if (distance < (particles->sml[bodyIndex + offset] * particles->sml[bodyIndex + offset])){ // && distance < (ghosts->sml[iGhost] * ghosts->sml[iGhost]))
                            // notice the interacting ghost particles
                            particles->nnlGhost[(bodyIndex + offset) * MAX_NUM_GHOST_INTERACTIONS + numInteractions] = iGhost;
                            numInteractions++;

                            if (numInteractions > MAX_NUM_GHOST_INTERACTIONS) {
                                // Issue massage
                                cudaTerminate("numInteractions = %i > MAX_NUM_GHOST_INTERACTIONS = %i\n", numInteractions, MAX_NUM_GHOST_INTERACTIONS);
                            }
                        }
                    }
                }
                // notice the number of Ghost interaction for each particle
                particles->noiGhost[bodyIndex + offset] = numInteractions;
            }

        }

        real Launch::fixedRadiusNNGhost_bruteForce(Particles *particles, Particles *ghosts) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNNGhost_bruteForce, *particles, *ghosts);
        }
    }
}