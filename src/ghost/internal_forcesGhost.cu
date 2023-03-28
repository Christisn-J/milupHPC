#include "../../include/ghost/densityGhost.cuh"

class densityGhost {

};

namespace SPH {

    namespace Kernel {

        __global__ void addAccelerateGhost(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles){

        }

        real Launch::addAccelerateGhost(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::addAccelerateGhost, kernel, tree, particles, interactions, numParticles);
        }

    }
}
