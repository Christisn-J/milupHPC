#include "parameter.h"
#ifndef PERIODIC_BOUNDARIES
#error "PERIODIC_BOUNDARIES is not defined!"
#endif

#if PERIODIC_BOUNDARIES
#include "../../include/ghost/addGhostDensity.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void addDensity_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts){
            /// debuge
            int MODPARTICLE = 5050;

            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            
            real W, Wj, dx[DIM], dWdx[DIM], dWdr;
            int ip;

            // iterate over all particles
            for(offset = 0; (bodyIndex + offset) < *particles->numParticles; offset += stride){
                /*
                /// debug
                if((bodyIndex + offset)%1000 == 0){
                    printf("(?)[DEBUG]    > particle: %i\n", (bodyIndex + offset));
                }
                */
                
                // iterate over all possible ghost particles
                for (int iGhost =0; iGhost< particles->noiGhost[(bodyIndex + offset)]; ++iGhost) {
                    ip = particles->nnlGhost[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + iGhost];

                        // calculate dimensionless the difference
                        dx[0] = particles->x[bodyIndex + offset] - ghosts->x[ip];
#if DIM > 1
                        dx[1] = particles->y[bodyIndex + offset] - ghosts->y[ip];
#if DIM == 3
                        dx[2] = particles->z[bodyIndex + offset] - ghosts->z[ip];
#endif // 3D
#endif // 2D

                    // Attention: initilase gohst particles mass (m)
                    kernel(&W, dWdx, &dWdr, dx, particles->sml[(bodyIndex + offset)]);
                    particles->rho[(bodyIndex + offset)] += ghosts->massGhosts[ip]*W;

                    /// debuge
                    if((bodyIndex + offset)%MODPARTICLE == 0){
                        printf("(?)[DEBUG]    > particle: %i \t >mass: %f \t >W: %f\n", (bodyIndex + offset), ghosts->massGhosts[ip], W);
                    }
                
                }
            }
        }

        real Launch::addDensity_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts) {
            ExecutionPolicy executionPolicy(1,1);
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::addDensity_Ghost, kernel, particles, ghosts);
        }
    }
}
#endif