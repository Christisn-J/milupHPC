#if PERIODIC_BOUNDARIES
#include "../../include/ghost/findGhostNN.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void fixedRadiusNNGhost_bruteForce(Particles *particles, IntegratedParticles *ghosts){
            /// debuge
            int MODPARTICLE = 5050;

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;

            real dx, dy, dz;
            real distance;
            int numInteractions;

            // iterate over all particles
            for(integer offset = 0; (bodyIndex + offset)<*(particles->numParticles); offset += stride){
                numInteractions = 0;
                
                /*
                /// debug
                if ((bodyIndex + offset)%1000 == 0){
                    printf("(?)[DEBUG]    > particle: %i \t noiG: %i\n", bodyIndex, numInteractions);
                }
                */
/// TODO: optimization search: ghost into tree or generate own tree for ghosts         
                // iterate over all possible ghost particles, not only (ghosts->numGhosts)
                for (int iGhost=0; iGhost< *(particles->numParticles)*MAX_GHOSTS_PER_PARTICLE; ++iGhost) {
                    

                    // sorting out all unnecessary ghosts
                    if(ghosts->usedGhosts[iGhost] != 0 ){
                        continue;
                    }

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
                            
                            /// debuge
                            if((bodyIndex + offset)%MODPARTICLE == 0){
                                printf("(?)[DEBUG]     > particle: %i (%.2f | %.2f)\n",(bodyIndex + offset), particles->x[bodyIndex + offset] ,particles->y[bodyIndex + offset]);
                            }
                            // Issue massage
                            cudaTerminate("numInteractions = %i > MAX_NUM_GHOST_INTERACTIONS = %i\n", numInteractions, MAX_NUM_GHOST_INTERACTIONS);
                        }
                    }
                }
                // notice the number of Ghost interaction for each particle
                particles->noiGhost[bodyIndex + offset] = numInteractions;

                /// debuge 
                if((bodyIndex + offset)%MODPARTICLE == 0)// numInteractions != 0 && 
                {
                    printf("(?)[DEBUG]    > particle: %i \t >noiG: %i\n", bodyIndex, numInteractions);
                }
                
                
            }

        }

        real Launch::fixedRadiusNNGhost_bruteForce(Particles *particles, IntegratedParticles *ghosts) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNNGhost_bruteForce, particles, ghosts);
        }
    }
}
#endif