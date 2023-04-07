#include "../../include/ghost/addGhostForces.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void addGhostForces(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticleHandler *ghosts){
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            
            real W, Wj, dx[DIM], dWdx[DIM], dWdr;
            int ip;

            for(integer offset = 0; (bodyIndex + offset) < particles->numParticles; offset += stride){
                for(int n=0; n<particles->noiGhost[(bodyIndex + offset)]; ++n){
                    int ip = particles->nnlGhost[n+(bodyIndex + offset)*MAX_GHOST_INTERACTIONS];

                    if (r == 0){
                        // no total share because of r := |r_i-r_n| multiplier
                        continue;
                    }

                        // calculate dimensionless the difference
                        dx[0] = particles->x[bodyIndex + offset] - ghosts->x[ip];
#if DIM > 1
                        dx[1] = particles->y[bodyIndex + offset] - ghosts->y[ip];
#if DIM == 3
                        dx[2] = particles->z[bodyIndex + offset] - ghosts->z[ip];
#endif // 3D
#endif // 2D

                    kernel(&W, dWdx, &dWdr, dx, particles->sml[(bodyIndex + offset)]);
                    particles->ax[(bodyIndex + offset)] += -ghosts->m[ip]*(particles->p[(bodyIndex + offset)]/(particles->rho[(bodyIndex + offset)]*particles->rho[(bodyIndex + offset)]) + ghosts->p[ip]/(ghosts->rho[i]*ghosts->rho[i]))*dWdr* dx[0];
#if DIM > 1
                    particles->ay[(bodyIndex + offset)] += -ghosts->m[ip]*(particles->p[(bodyIndex + offset)]/(particles->rho[(bodyIndex + offset)]*particles->rho[(bodyIndex + offset)]) + ghosts->p[ip]/(ghosts->rho[i]*ghosts->rho[i]))*dWdr* dx[1];
#if DIM == 3
                    particles->az[(bodyIndex + offset)] += -ghosts->m[ip]*(particles->p[(bodyIndex + offset)]/(particles->rho[(bodyIndex + offset)]*particles->rho[(bodyIndex + offset)]) + ghosts->p[ip]/(ghosts->rho[i]*ghosts->rho[i]))*dWdr* dx[2];
#endif // 3D
#endif // 2D
                }

            }
        }

        real Launch::addGhostForces(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticleHandler *ghosts){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::addGhostForces, kernel, particles, ghosts);
        }

    }
}
