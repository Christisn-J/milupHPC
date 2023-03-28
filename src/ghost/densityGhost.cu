#include "../../include/ghost/densityGhost.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void addDensityGhost(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles){

        }

        real Launch::addDensityGhost(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateDensity, kernel, tree, particles, interactions, numParticles);
        }
    }
}

#endif //MILUPHPC_DENSITYGHOST_CUH
