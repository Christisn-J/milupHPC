/**
 * @file accelerateGhost.cuh
 * @brief add SPH accelerate caused by Ghost.
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_ACCELERATEGHOST_CUH
#define MILUPHPC_ACCELERATEGHOST_CUH

#include "../particles.cuh"
#include "../sph/sph.cuh"
#include "../sph/kernel.cuh"
#include "../particle_handler.h"
#include "../parameter.h"


class densityGhost {

};

namespace SPH {

    namespace Kernel {
        /**
         * @brief  calculates the influence of the ghost particles and add the result on top of the particles acceleration \f$ \a \f$.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::addGhostForces()
         *
         * Here, similar to the ghost density calculation (::SPH::Kernel::addGhostDensity()), all the interaction partner
         * particles are iterated for each particle and those contributions are added.
         * 
         * 
/// TODO: Document formular
         *
         * @param kernel SPH smoothing kernel
         * @param particles Particles class instance
         * @param ghosts IntegratedParticles class instance
         */

        __global__ void addForces_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::addGhostDensity().
             * 
             * @param kernel SPH smoothing kernel
             * @param particles Particles class instance
             * @param ghosts IntegratedParticles class instance
             */
            real addForces_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts);
        }

    }
}

#endif //MILUPHPC_ACCELERATEGHOST_CUH
