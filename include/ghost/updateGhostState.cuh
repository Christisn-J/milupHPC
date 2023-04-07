/**
 * @file updateGhostState.cuh
 * @brief update all ghost variables.
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
        /**
         * @brief  compare for each ghost particle all values ​​with original particle.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::updateGhostState()
         *
         * Update all values ​​related to the origin particle.
         *
         * @param particles Particles class instance
         * @param ghosts IntegratedParticleHandler class instance
         */

        __global__ void updateGhostState(Particles *particles, IntegratedParticleHandler *ghosts);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::updateGhostState().
             * 
             * @param particles Particles class instance
             * @param ghosts IntegratedParticleHandler class instance
             */
            real updateGhostState(Particles *particles, IntegratedParticleHandler *ghosts);
        }
    }
}
#endif // MILUPHPC_UPDATEGHOSTSTATE_CUH 