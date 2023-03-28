#include "../../include/ghost/updateGhostState.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void updateGhostState(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes){

        }

        real Launch::updateGhostState(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateDensity, kernel, tree, particles, interactions, numParticlesLocal, numParticles, numNodes);
        }
    }
}