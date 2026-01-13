#include "../../include/sph/pressure.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"


namespace EOS {
    __device__ void polytropicGas(Material *materials, Particles *particles, int index) {
        //printf("polytropicGas...\n");
        particles->p[index] = materials[particles->materialId[index]].eos.polytropic_K *
                              pow(particles->rho[index], materials[particles->materialId[index]].eos.polytropic_gamma);
        //if (true /*particles->p[index] > 0.*/) {
        //    printf("pressure: p[%i] = %f, rho[%i] = %f, polyTropic_K = %f, polytropic_gamma = %f\n", index,
        //           particles->p[index], index, particles->rho[index], materials[particles->materialId[index]].eos.polytropic_K,
        //           materials[particles->materialId[index]].eos.polytropic_K);
        //}
        //return;
    }

    __device__ void murnaghan(Material *materials, Particles *particles, int index) {
        particles->p[index] = (materials[particles->materialId[index]].eos.bulk_modulus / materials[particles->materialId[index]].eos.n) *
                              (pow(particles->rho[index] / materials[particles->materialId[index]].eos.rho_0, materials[particles->materialId[index]].eos.n) - 1.0);
        //return;
    }

    __device__ void tillotson(Material *materials, Particles *particles, int index,
                              double rho, double e, double &eta, double &mu, double &p1, double &p2) {
        int matId = particles->materialId[index];
        eta = rho / materials[matId].eos.rho_0;
        mu = eta - 1.0;

        double omega0 = e / (materials[matId].eos.E_0 * pow(eta, 2.0)) + 1.0;

        // Check if the energy is within the compressed region (e <= E_iv)
        if (e <= materials[matId].eos.E_iv) {
            particles->p[index] =
                    (materials[matId].eos.till_a
                     + materials[matId].eos.till_b / omega0)
                    * rho * e
                    + materials[matId].eos.till_A * mu
                    + materials[matId].eos.till_B * pow(mu, 2.0);
        }
            // Check if the energy is within the expanded region (e >= E_cv)
        else if (e >= materials[matId].eos.E_cv) {
            particles->p[index] =
                    materials[matId].eos.till_a * rho * e
                    + ((materials[matId].eos.till_b * rho * e / omega0)
                       + materials[matId].eos.till_A * mu * exp(-materials[matId].eos.till_beta * ((materials[matId].eos.rho_0 / rho) - 1.0)))
                      * exp(-materials[matId].eos.till_alpha * pow(((materials[matId].eos.rho_0 / rho) - 1.0), 2.0));

        }
            // Partial vaporization region (E_iv < e < E_cv)
        else if (e > materials[matId].eos.E_iv && e < materials[matId].eos.E_cv) {
            double weight = (e - materials[matId].eos.E_iv) /
                            (materials[matId].eos.E_cv - materials[matId].eos.E_iv);

            p1 = (materials[matId].eos.till_a
                  + materials[matId].eos.till_b / omega0)
                 * rho * e
                 + materials[matId].eos.till_A * mu
                 + materials[matId].eos.till_B * pow(mu, 2.0);

            p2 = materials[matId].eos.till_a * rho * e
                 + ((materials[matId].eos.till_b * rho * e / omega0)
                    + materials[matId].eos.till_A * mu * exp(-materials[matId].eos.till_beta * ((materials[matId].eos.rho_0 / rho) - 1.0)))
                   * exp(-materials[matId].eos.till_alpha * pow(((materials[matId].eos.rho_0 / rho) - 1.0), 2.0));

            particles->p[index] = (1.0 - weight) * p1 + weight * p2;
        } else {
            printf("\nDeep trouble in pressure.\nenergy[%d] = %e\nE_iv = %e, E_cv = %e\n\n", index, e, materials[matId].eos.E_iv, materials[matId].eos.E_cv);
            particles->p[index] = 0.0;
        }
        //return;
    }

    __device__ void isothermalGas(Material *materials, Particles *particles, int index) {
        //printf("isothermalGas...\n");
        particles->p[index] = 41255.407 * particles->rho[index];
        //return;
    }

    __device__ void idealGas(Material *materials, Particles *particles, int index) {
        //printf("idealGas...\n");
        //if (index % 1000 == 0) {
        //    printf("polytropic gamma: %e\n", materials[particles->materialId[index]].eos.polytropic_gamma);
        //}
        particles->p[index] = (materials[particles->materialId[index]].eos.polytropic_gamma - 1) *
                              particles->rho[index] * particles->e[index];
        if (particles->p[index] < 0) {
            printf("negative pressure! p[%i] = %e, rho = %e, e = %e\n", index, particles->p[index], particles->rho[index], particles->e[index]);
        }
        //particles->p[index] = particles->cs[index] * particles->cs[index] * particles->rho[index];
        //return;
    }

    __device__ void locallyIsothermalGas(Material *materials, Particles *particles, int index) {
        //printf("locallyIsothermalGas...\n");
        particles->p[index] = particles->cs[index] * particles->cs[index] * particles->rho[index];
        //return;
    }
}

namespace SPH {
    namespace Kernel {
        __global__ void calculatePressure(Material *materials, Particles *particles, int numParticles) {

            register int i, inc;
            register double eta, e, rho, mu, p1, p2;
            int i_rho, i_e;
            double pressure;

            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

                pressure = 0.0;

                //printf("calculatePressure: %i\n", materials[particles->materialId[i]].eos.type);
                switch (materials[particles->materialId[i]].eos.type) {
                    case EquationOfStates::EOS_TYPE_POLYTROPIC_GAS: {
                        ::EOS::polytropicGas(materials, particles, i);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_MURNAGHAN: {
                        ::EOS::murnaghan(materials, particles, i);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_TILLOTSON: {
                        rho = particles->rho[i];
                        e = particles->e[i];
                        ::EOS::tillotson(materials, particles, i, rho, e, eta, mu, p1, p2);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_ISOTHERMAL_GAS: {
                        ::EOS::isothermalGas(materials, particles, i);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_IDEAL_GAS: {
                        ::EOS::idealGas(materials, particles, i);
                    }
                        break;
                    case EquationOfStates::EOS_TYPE_LOCALLY_ISOTHERMAL_GAS: {
                        ::EOS::locallyIsothermalGas(materials, particles, i);
                    }
                        break;
                    default:
                        printf("not implemented!\n");
                }

            }
        }

        real Launch::calculatePressure(Material *materials, Particles *particles, int numParticles) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculatePressure, materials,
                                particles, numParticles);
        }
    }
}

