#include "../../include/materials/material.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Material::Material() {

}

CUDA_CALLABLE_MEMBER Material::~Material() {

}

CUDA_CALLABLE_MEMBER void Material::info() {
    // TODO: Maybe add switch or #if, #else, ... to only print relevant paramters
    printf("-------------------------- Material Information --------------------------\n");
    printf("Material: ID                                        = %i\n", ID);
    printf("Material: interactions                              = %i\n", interactions);
    // Artificial Viscosity Parameters
    printf("Material: alpha                                     = %f\n", artificialViscosity.alpha);
    printf("Material: beta                                      = %f\n", artificialViscosity.beta);
    // EOS Parameters
    printf("Material: eos: type                                 = %i\n", eos.type);
    printf("Material: eos: polytropic_K                         = %f\n", eos.polytropic_K);
    printf("Material: eos: polytropic_gamma                     = %f\n", eos.polytropic_gamma);
    printf("Material: eos: rho0                                 = %f\n", eos.rho_0);
    printf("Material: eos: bulk_modulus                         = %f\n", eos.bulk_modulus);
    printf("Material: eos: n                                    = %f\n", eos.n);
    printf("Material: eos: shear_modulus                        = %f\n", eos.shear_modulus);
    printf("Material: eos: young_modulus                        = %f\n", eos.young_modulus);
    printf("Material: eos: yield_stress                         = %f\n", eos.yield_stress);
    // Artificial Stress Parameters
#if ARTIFICIAL_STRESS
    printf("Material: artificial Stress: exponent tensor        = %f\n", artificialStress.exponent_tensor);
    printf("Material: artificial Stress: epsilon                = %f\n", artificialStress.epsilon_stress);
    printf("Material: artificial Stress: mean particle distance = %f\n", artificialStress.mean_particle_distance);
#endif
    // Tillotson EOS Parameters
    printf("Material: eos: till_A                                = %f\n", eos.till_A);
    printf("Material: eos: till_B                                = %f\n", eos.till_B);
    printf("Material: eos: E_0                                   = %f\n", eos.E_0);
    printf("Material: eos: E_iv                                  = %f\n", eos.E_iv);
    printf("Material: eos: E_cv                                  = %f\n", eos.E_cv);
    printf("Material: eos: till_a                                = %f\n", eos.till_a);
    printf("Material: eos: till_b                                = %f\n", eos.till_b);
    printf("Material: eos: till_alpha                            = %f\n", eos.till_alpha);
    printf("Material: eos: till_beta                             = %f\n", eos.till_beta);
    printf("Material: eos: rho_limit                             = %f\n", eos.rho_limit);
    printf("Material: eos: cs_limit                              = %f\n", eos.cs_limit);
    // TODO: add other parameters
    printf("-------------------------------------------------------------------------\n");
}

namespace MaterialNS {
    namespace Kernel {
        __global__ void info(Material *material) {
            material->info();
        }

        void Launch::info(Material *material) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::MaterialNS::Kernel::info, material);
        }
    }
}


CUDA_CALLABLE_MEMBER ArtificialViscosity::ArtificialViscosity() : alpha(0.0), beta(0.0) {

}

CUDA_CALLABLE_MEMBER ArtificialViscosity::ArtificialViscosity(real alpha, real beta) : alpha(alpha), beta(beta) {

}
// TODO: Add Artificial Stress? not necessary

// TODO: Modify? for other EOS, not necessary
CUDA_CALLABLE_MEMBER EqOfSt::EqOfSt() : type(0), polytropic_K(0.), polytropic_gamma(0.) {

}
// TODO: Modify? for other EOS, not necessary
CUDA_CALLABLE_MEMBER EqOfSt::EqOfSt(int type, real polytropic_K, real polytropic_gamma) : type(type),
                                                                                          polytropic_K(polytropic_K), polytropic_gamma(polytropic_gamma) {

}

