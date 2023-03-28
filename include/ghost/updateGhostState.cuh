/**
 * @file updateGhostState.cuh
 * @brief .
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_UPDATEGHOSTSTATE_CUH
#define MILUPHPC_UPDATEGHOSTSTATE_CUH

#include "../particles.cuh"
#include "../sph/sph.cuh"

namespace SPH {

    namespace Kernel {
        __global__ void updateGhostState(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes);

        namespace Launch {
            real updateGhostState(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes);
        }
    }
}
#endif // MILUPHPC_UPDATEGHOSTSTATE_CUH 