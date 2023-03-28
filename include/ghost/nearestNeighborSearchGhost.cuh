/**
 * @file nearestNeighborSearchGhost.cuh
 * @brief creat ghost particles.
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_NEARESTNEIGHBORSEARCHGHOST_CUH
#define MILUPHPC_NEARESTNEIGHBORSEARCHGHOST_CUH

namespace SPH {

    namespace Kernel {
        __global__ void fixedRadiusNNGhost_bruteForce(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes);

        namespace Launch {
            real fixedRadiusNNGhost_bruteForce(Tree *tree, Particles *particles, integer *interactions,integer numParticlesLocal, integer numParticles, integer numNodes);
        }
    }
}
#endif // MILUPHPC_NEARESTNEIGHBORSEARCHGHOST_CUH 