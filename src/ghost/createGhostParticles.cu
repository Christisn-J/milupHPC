#include "../../include/ghost/createGhostParticles.cuh"

namespace SPH {

    namespace Kernel {
        __global__ void createGhostParticles(Tree *tree, Particles *particles,integer numParticlesLocal, integer numParticles, integer numNodes){

        }

        Particles Launch::createGhostParticles(Tree *tree, Particles *particles,integer numParticlesLocal, integer numParticles, integer numNodes){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::createGhostParticles, tree, particles, numParticlesLocal, numParticles, numNodes);
        }
    }
}
#endif // MILUPHPC_CREATGHOSTPARTICLES_CUH 