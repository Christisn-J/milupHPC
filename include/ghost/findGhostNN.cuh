/**
 * @file nearestNeighborSearchGhost.cuh
 * @brief find nearest ghost neighbor
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_NEARESTNEIGHBORSEARCHGHOST_CUH
#define MILUPHPC_NEARESTNEIGHBORSEARCHGHOST_CUH

#include "../particles.cuh"
#include "../sph/sph.cuh"

namespace SPH {

    namespace Kernel {
        /**
         * @brief  for each particle finding the ghosts interaction.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::fixedRadiusNNGhost_bruteForce()
         *
         * Checks for each particle if a ghost particle is near enough \f[|x_i -x_g| < h]\f and remembers it if necessary.
         *
         * @param [in] particles Particles class instance
         * @param [in] ghosts IntegratedParticleHandler class instance
         * @param [out] nnlGhost a ghost nearest neighbor list
         * @param [out] noiGhost number of ghost interactions
         */

        __global__ void fixedRadiusNNGhost_bruteForce(Particles *particles, Particles *ghosts);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::fixedRadiusNNGhost_bruteForce().
             * 
             * @param particles Particles class instance
             * @param ghosts IntegratedParticleHandler class instance
             */
            real fixedRadiusNNGhost_bruteForce(Particles *particles, Particles *ghosts);
        }
    }
}
#endif // MILUPHPC_NEARESTNEIGHBORSEARCHGHOST_CUH 