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
#include "../particle_handler.h"
#include "../parameter.h"

namespace SPH {

    namespace Kernel {
        /**
         * @brief  calculates the influence of the ghost particles and add the result on top of the particles acceleration \f$ \a \f$.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::createGhostParticles()
         *
         * In this function it is checked whether a particle feels the conditions for a ghost particle. 
         * If so, up to \f$2*Dim-1$\f ghost particles are formed from each particle.
         * In the end, the total number of ghost particles is stored in the ghost class.
         * 
         *\f[
         * Conditions for a ghost particle is teh following:
         * \begin{equation}
	        domain_{max} - h < x_{i} <= domain_{min} + h
         *  \end{equation}
         *\f]
         * Here \f$h$\f is the smoothing length.
         * It says, if a particle is inside the domain and close enougth to the domain boundary. 
         *
         * @param [in] particles Particles class instance
         * @param [in] tree Tree class instance
         * @param ghosts IntegratedParticleHandler class instance
         * @param [out] numGhosts total number of ghost particles
         */

        __global__ void createGhostsPeriodic(Tree *tree, Particles *particles, IntegratedParticles *ghosts, integer *iGhost);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::addGhostDensity().
             * 
             * @param particles Particles class instance
             * @param tree Tree class instance
             * @param ghosts IntegratedParticleHandler class instance
             */
            real createGhostsPeriodic(Tree *tree, Particles *particles, IntegratedParticles *ghosts, integer *iGhost);
        }
    }
}
#endif // MILUPHPC_CREATGHOSTPARTICLES_CUH 