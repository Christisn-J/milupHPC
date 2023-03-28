/**
 * @file densityGhost.cuh
 * @brief add SPH density caused by ghost.
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_DENSITYGHOST_CUH
#define MILUPHPC_DENSITYGHOST_CUH

#include "../particles.cuh"
#include "../sph/sph.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void addDensityGhost(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles);

        namespace Launch {
            real addDensityGhost(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles);
        }

    }
}

#endif //MILUPHPC_DENSITYGHOST_CUH
