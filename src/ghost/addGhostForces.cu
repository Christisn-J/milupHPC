#if PERIODIC_BOUNDARIES
#include "../../include/ghost/addGhostForces.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void addForces_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts){
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            
            real W, Wj, dx[DIM], dWdx[DIM], dWdr;
            int ip;

            for(integer offset = 0; (bodyIndex + offset) < *particles->numParticles; offset += stride){
                for(int n=0; n<particles->noiGhost[(bodyIndex + offset)]; ++n){
                    int ip = particles->nnlGhost[n+(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE];

                        // calculate dimensionless the difference
                        dx[0] = particles->x[bodyIndex + offset] - ghosts->x[ip];
#if DIM > 1
                        dx[1] = particles->y[bodyIndex + offset] - ghosts->y[ip];
#if DIM == 3
                        dx[2] = particles->z[bodyIndex + offset] - ghosts->z[ip];
#endif // 3D
#endif // 2D

                    if (dx[0] == 0
#if DIM > 1
                        && dx[1] == 0 
#if DIM == 3
                        && dx[2] == 0
#endif // 3D
#endif // 2D
                        ){
                        // no total share because of r := |r_i-r_n| multiplier
                        continue;
                    }

                    kernel(&W, dWdx, &dWdr, dx, particles->sml[(bodyIndex + offset)]);
                    
                    particles->ax[(bodyIndex + offset)] += -ghosts->massGhosts[ip]*(particles->p[(bodyIndex + offset)]/(particles->rho[(bodyIndex + offset)]*particles->rho[(bodyIndex + offset)]) + ghosts->p[ip]/(ghosts->rho[ip]*ghosts->rho[ip]))*dWdr* dx[0];
#if DIM > 1
                    particles->ay[(bodyIndex + offset)] += -ghosts->massGhosts[ip]*(particles->p[(bodyIndex + offset)]/(particles->rho[(bodyIndex + offset)]*particles->rho[(bodyIndex + offset)]) + ghosts->p[ip]/(ghosts->rho[ip]*ghosts->rho[ip]))*dWdr* dx[1];
#if DIM == 3
                    particles->az[(bodyIndex + offset)] += -ghosts->massGhosts[ip]*(particles->p[(bodyIndex + offset)]/(particles->rho[(bodyIndex + offset)]*particles->rho[(bodyIndex + offset)]) + ghosts->p[ip]/(ghosts->rho[(ip]*ghosts->rho[ip]))*dWdr* dx[2];
#endif // 3D
#endif // 2D
                }

            }
        }

        real Launch::addForces_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts){
            ExecutionPolicy executionPolicy(1,1);
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::addForces_Ghost, kernel, particles, ghosts);
        }

    }
}
#endif
