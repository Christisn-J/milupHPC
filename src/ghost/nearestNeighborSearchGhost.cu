#include "../../include/ghost/nearestNeighborSearchGhost.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void fixedRadiusNNGhost_bruteForce(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes){
            
        }

        real Launch::fixedRadiusNNGhost_bruteForce(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::fixedRadiusNNGhost_bruteForce, tree, particles, interactions, numParticlesLocal, numParticles, numNodes);
        }
    }
}