/**
 * @file updateGhostState.cuh
 * @brief update all ghost variables.
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_UPDATEGHOSTSTATE_CUH
#define MILUPHPC_UPDATEGHOSTSTATE_CUH

#if PERIODIC_BOUNDARIES
#include "../particles.cuh"
#include "../sph/sph.cuh"
#include "../parameter.h"

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
         * @param ghosts IntegratedParticles class instance
         */

        __global__ void updateGhosts(Particles *particles, IntegratedParticles *ghosts);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::updateGhostState().
             * 
             * @param particles Particles class instance
             * @param ghosts IntegratedParticles class instance
             */
            real updateGhosts(Particles *particles, IntegratedParticles *ghosts);
        }
    }
}
#endif
#endif // MILUPHPC_UPDATEGHOSTSTATE_CUH 