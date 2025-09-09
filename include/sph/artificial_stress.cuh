/**
 * @file artificial_stress.cuh
 * @brief Artificial Stress.
 *
 * Calculate the artificial stress tensor for each particle
 *
 * @author Anne Vera Jeschke
 * @bug no known bugs
 */
#ifndef MILUPHPC_ARTIFICIAL_STRESS_CUH
#define MILUPHPC_ARTIFICIAL_STRESS_CUH

#include "../constants.h"
#include "../particles.cuh"
#include "../materials/material.cuh"

namespace SPH {
    /**
     * @brief Calculate and set the artificial stress for one particle
     *
     * calculate and set the artificial stress tensor \f$R^{ij}\f$ for each particle
     * \f{align}{
            R^{ij} = \begin{cases}
            -\epsilon \sigma^{ij}_a & \text{for } \sigma^{ij} > 0 \\
            0 & \text{otherwise} \\
            \end{cases}
     * \f}
     *
     * @param materials Material class instance
     * @param sigma stress
     * @param[out] R artificial stress
     * @param matId matId of corresponding particles
     */
    __device__ void calcArtificialStress(Material *materials, real sigma[DIM][DIM], real R[DIM][DIM], int matId);

}


#endif //MILUPHPC_STRESS_CUH