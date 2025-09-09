/**
 * @file stress.cuh
 * @brief Stress.
 *
 * Calculate the Stress tensor from the pressure and the deviatoric stress for solids.
 *
 * @author Michael Staneker, Anne Vera Jeschke
 * @bug no known bugs
 */
#ifndef MILUPHPC_STRESS_CUH
#define MILUPHPC_STRESS_CUH

#include "../constants.h"
#include "../particles.cuh"

namespace SPH {
        /**
           * @brief Calculate and Set the stress for one particle p
           *
           * calculate and set stress tensor \f$\sigma^{ij}\f$ for one particle from the deviatoric stress \f$S^{ij}\f$ and the pressure \f$p\f$
           * \f{align}{
                  \sigma^{ij = -p\delta^{ij} + S^{ij}
           * \f}
           *
           * @param[in] particles Particles class instance
           * @param[out] sigma stress tensor for particle p
           * @param[in] p particle index for which stress tensor is calculated
           */
        __device__ void calcStress(Particles *particles, real sigma[DIM][DIM], int p);
}


#endif //MILUPHPC_STRESS_CUH
