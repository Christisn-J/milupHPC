/**
 * @file createGhostParticles.cuh
 * @brief creat ghost particles.
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_CREATGHOSTPARTICLES_CUH
#define MILUPHPC_CREATGHOSTPARTICLES_CUH

#include "../particles.cuh"
#include "../sph/sph.cuh"

namespace SPH {

    namespace Kernel {
        __global__ void createGhostParticles(Tree *tree, Particles *particles,integer numParticlesLocal, integer numParticles, integer numNodes);

        namespace Launch {
            Particles createGhostParticles(Tree *tree, Particles *particles,integer numParticlesLocal, integer numParticles, integer numNodes);
        }
    }
}
#endif // MILUPHPC_CREATGHOSTPARTICLES_CUH 