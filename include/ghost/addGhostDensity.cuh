/**
 * @file addGhostDensity.cuh
 * @brief add SPH density caused by ghost.
 *
 * @author Christian Jetter
 * @bug no known bugs
 */
#ifndef MILUPHPC_DENSITYGHOST_CUH
#define MILUPHPC_DENSITYGHOST_CUH

#include "../parameter.h"
#if PERIODIC_BOUNDARIES
#include "../particle_handler.h"
#include "../particles.cuh"
#include "../sph/sph.cuh"
#include "../sph/kernel.cuh"


namespace SPH {

    namespace Kernel {
        /**
         * @brief calculates the influence of the ghost particles and add the result on top of the particles density \f$ \rho \f$.
         *
         * > Corresponding wrapper function: ::Physics::Kernel::Launch::addGhostDensity()
         *
         * In order to compute influence on the density from the ghost particles, all ghost interaction partners for each particle are iterated and those masses
         * taken into account weighted with the smoothing kernel.
         * 
         * \f[
         * The density is given by the kernel sum
         *  \begin{equation}
	            \rho_a = \sum_{b} m_b W_{ab} \, .
         *  \end{equation}
         * \f]
         * 
         * @param kernel SPH smoothing kernel
         * @param particles Particles class instance
         * @param ghosts IntegratedParticles class instance
         */

        __global__ void addDensity_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::addGhostDensity().
             * 
             * @param kernel SPH smoothing kernel
             * @param particles Particles class instance
             * @param ghosts IntegratedParticles class instance
             */
            real addDensity_Ghost(::SPH::SPH_kernel kernel, Particles *particles, IntegratedParticles *ghosts);
        }

    }
}
#endif
#endif //MILUPHPC_DENSITYGHOST_CUH
