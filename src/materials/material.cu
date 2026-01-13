#include "../../include/materials/material.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

/**
 * @brief Default constructor.
 */
CUDA_CALLABLE_MEMBER Material::Material() {}

/**
 * @brief Destructor.
 *
 * This destructor is intentionally left empty because all member variables
 * of Material are either primitive types or trivially destructible structures.
 */
CUDA_CALLABLE_MEMBER Material::~Material() {}

/**
 * @brief Print material information to console.
 *
 * Outputs all material properties, including optional parameters
 * depending on compile-time flags (ARTIFICIAL_VISCOSITY, ARTIFICIAL_STRESS, PLASTICITY).
 *
 * @note TODO: Extend this function to include any additional material parameters.
 */
CUDA_CALLABLE_MEMBER void Material::info() {
//TODO: Consider adding conditionals or #if/#else to print only relevant parameters.
    printf("-------------------------- Material Information --------------------------\n");
#if CHECK_MATERIAL
    printf("Material: integer                                    = %i\n", dummy_int);
    printf("Material: real                                       = %f\n", dummy_real);
#endif
    printf("Material: ID                                        = %i\n", ID);
    printf("Material: interactions                              = %i\n", interactions);

#if ARTIFICIAL_VISCOSITY
    // Artificial Viscosity Parameters
    printf("Material: artificial Viscosity: alpha               = %f\n", artificialViscosity.alpha);
    printf("Material: artificial Viscosity: beta                = %f\n", artificialViscosity.beta);
#endif

#if ARTIFICIAL_STRESS
    // Artificial Stress Parameters
    printf("Material: artificial Stress: exponent tensor        = %f\n", artificialStress.exponent_tensor);
    printf("Material: artificial Stress: epsilon                = %f\n", artificialStress.epsilon_stress);
    printf("Material: artificial Stress: mean particle distance = %f\n", artificialStress.mean_particle_distance);
#endif

#if PLASTICITY
    // Plasticity Parameters
    printf("Material: plasticity: yield_stress                  = %f\n", plasticity.yield_stress);
//    printf("Material: eos: yield_stress                         = %f\n", eos.yield_stress);
#endif

    // Equation of State Parameters
    printf("Material: eos: type                                 = %i\n", eos.type);
    printf("Material: eos: rho0                                 = %f\n", eos.rho_0);

    printf("Material: eos: n                                    = %f\n", eos.n);
    printf("Material: eos: shear_modulus                        = %f\n", eos.shear_modulus);
    printf("Material: eos: young_modulus                        = %f\n", eos.young_modulus);

    printf("Material: eos: polytropic_K                         = %f\n", eos.polytropic_K);
    printf("Material: eos: polytropic_gamma                     = %f\n", eos.polytropic_gamma);
    printf("Material: eos: bulk_modulus                         = %f\n", eos.bulk_modulus);
    printf("Material: eos: till_A                               = %f\n", eos.till_A);
    printf("Material: eos: till_B                               = %f\n", eos.till_B);
    printf("Material: eos: E_0                                  = %f\n", eos.E_0);
    printf("Material: eos: E_iv                                 = %f\n", eos.E_iv);
    printf("Material: eos: E_cv                                 = %f\n", eos.E_cv);
    printf("Material: eos: till_a                               = %f\n", eos.till_a);
    printf("Material: eos: till_b                               = %f\n", eos.till_b);
    printf("Material: eos: till_alpha                           = %f\n", eos.till_alpha);
    printf("Material: eos: till_beta                            = %f\n", eos.till_beta);
    printf("Material: eos: rho_limit                            = %f\n", eos.rho_limit);
    printf("Material: eos: cs_limit                             = %f\n", eos.cs_limit);

    printf("-------------------------------------------------------------------------\n");
}

namespace MaterialNS {
    namespace Kernel {
        /**
         * @brief CUDA kernel to call Material::info() on device.
         *
         * @param material Pointer to the Material instance on device.
         */
        __global__ void info(Material *material) {
            material->info();
        }

        /**
         * @brief Launches CUDA kernel to print material information.
         *
         * Uses custom ExecutionPolicy and cuda::launch utility.
         *
         * @param material Pointer to the Material instance on device.
         */
        void Launch::info(Material *material) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::MaterialNS::Kernel::info, material);
        }
    }
}
