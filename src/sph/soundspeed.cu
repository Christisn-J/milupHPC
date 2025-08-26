#include "../../include/sph/soundspeed.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void SPH::Kernel::initializeSoundSpeed(Particles *particles, Material *materials, int numParticles) {

    register int i, inc, matId;
    inc = blockDim.x * gridDim.x;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = particles->materialId[i];

        switch (materials[matId].eos.type) {
            case EquationOfStates::EOS_TYPE_POLYTROPIC_GAS: {
                particles->cs[i] = 0.0; // for gas this will be calculated each step by kernel calculateSoundSpeed
            }
                break;
            case EquationOfStates::EOS_TYPE_MURNAGHAN: {
                particles->cs[i] = cuda::math::sqrt(materials[matId].eos.bulk_modulus / materials[matId].eos.rho_0); // c_s constant
            }
                break;
            case EquationOfStates::EOS_TYPE_TILLOTSON: {
                particles->cs[i] = cuda::math::sqrt(materials[matId].eos.bulk_modulus / materials[matId].eos.rho_0);
            }
                break;
            case EquationOfStates::EOS_TYPE_ISOTHERMAL_GAS: {
                particles->cs[i] = 203.0; // this is pure molecular hydrogen at 10 K
                //if (i % 1000 == 0) {
                //    printf("cs[%i]: %e\n", i, particles->cs[i]);
                //}
#if !SI_UNITS
                particles->cs[i] /= 2.998e8; // speed of light
#endif
            }
                break;
            case EquationOfStates::EOS_TYPE_LOCALLY_ISOTHERMAL_GAS: {
                //TODO: initial sound speed for EOS_TYPE_LOCALLY_ISOTHERMAL_GAS?
            }
                break;
                //default:
                //    printf("not implemented!\n");
        }
    }

}

__global__ void SPH::Kernel::calculateSoundSpeed(Particles *particles, Material *materials, int numParticles) {

    register int i, inc, matId;
    int d;
    int j;
    double m_com;
    register double cs, rho, pressure, eta, omega0, z, cs_sq, cs_c_sq, cs_e_sq, Gamma_e, mu, y; //Gamma_c;
    int i_rho, i_e;

    inc = blockDim.x * gridDim.x;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        matId = particles->materialId[i];

        switch (materials[matId].eos.type) {
            case EquationOfStates::EOS_TYPE_POLYTROPIC_GAS: {
                particles->cs[i] = cuda::math::sqrt(materials[matId].eos.polytropic_K *
                                                    pow(particles->rho[i], materials[matId].eos.polytropic_gamma - 1.0));
            }
                break;
                //case EquationOfStates::EOS_TYPE_MURNAGHAN: {
                //     // do nothing since c_s is constant
                //} break;
            case EquationOfStates::EOS_TYPE_TILLOTSON: {
                // 	Translation Matrix
                // rho = particles->rho[i];
                // pressure = particles->p[i];
                // eta = rho / materials[matId].eos.rho_0;
                // mu = eta - 1.0;
                // omega0 = particles->e[i] / (materials[matId].eos.E_0 * eta * eta) + 1.0;
                // z = (1.0 - eta) / eta;

                if ((particles->rho[i] / materials[matId].eos.rho_0) >= 0.0 || particles->e[i] < materials[matId].eos.E_iv) {
                    if (particles->p[i] < 0.0 || (particles->rho[i] / materials[matId].eos.rho_0) < materials[matId].eos.rho_limit) particles->p[i] = 0.0;

                    cs_sq = materials[matId].eos.till_a * particles->e[i]
                            + (materials[matId].eos.till_b * particles->e[i]) /
                              ((particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                               (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0)) *
                              (3.0 * (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) - 2.0)
                            + (materials[matId].eos.till_a + 2.0 * materials[matId].eos.till_b * ((particles->rho[i] / materials[matId].eos.rho_0) - 1.0)) / particles->rho[i]
                            + particles->p[i] / (particles->rho[i] * particles->rho[i]) * (materials[matId].eos.till_a * particles->rho[i] + materials[matId].eos.till_b * particles->rho[i] /
                                                                                                                                             ((particles->e[i] / (materials[matId].eos.E_0 *
                                                                                                                                                                  (particles->rho[i] /
                                                                                                                                                                   materials[matId].eos.rho_0) *
                                                                                                                                                                  (particles->rho[i] /
                                                                                                                                                                   materials[matId].eos.rho_0)) + 1.0) *
                                                                                                                                              (particles->e[i] / (materials[matId].eos.E_0 *
                                                                                                                                                                  (particles->rho[i] /
                                                                                                                                                                   materials[matId].eos.rho_0) *
                                                                                                                                                                  (particles->rho[i] /
                                                                                                                                                                   materials[matId].eos.rho_0)) +
                                                                                                                                               1.0)));
                } else if (particles->e[i] > materials[matId].eos.E_cv) {
                    Gamma_e = materials[matId].eos.till_a
                              + materials[matId].eos.till_b /
                                (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                exp(-materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                    ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)));

                    cs_sq = (Gamma_e + 1.0) * particles->p[i] / particles->rho[i]
                            + materials[matId].eos.till_a / particles->rho[i]
                              * exp(-(materials[matId].eos.till_alpha * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) +
                                      materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                      ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0))))
                              * (1.0 + ((particles->rho[i] / materials[matId].eos.rho_0) - 1.0)) / ((particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0))
                              * (materials[matId].eos.till_alpha +
                                 2.0 * materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) -
                                 (particles->rho[i] / materials[matId].eos.rho_0))
                            + materials[matId].eos.till_b * particles->rho[i] * particles->e[i]
                              / ((particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                 (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                 (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0))
                              * exp(-materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                    ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)))
                              * (2.0 * materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                 (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) /
                                 materials[matId].eos.rho_0 + 1.0)
                              / (materials[matId].eos.E_0 * particles->rho[i])
                              * (2.0 * particles->e[i] - particles->p[i] / particles->rho[i]);
                } else {
                    Gamma_e = materials[matId].eos.till_a
                              + materials[matId].eos.till_b /
                                (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                exp(-materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                    ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)));

                    cs_e_sq = (Gamma_e + 1.0) * particles->p[i] / particles->rho[i]
                              + materials[matId].eos.till_a / particles->rho[i]
                                * exp(-(materials[matId].eos.till_alpha * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) +
                                        materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                        ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0))))
                                * (1.0 + ((particles->rho[i] / materials[matId].eos.rho_0) - 1.0)) /
                                ((particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0))
                                * (materials[matId].eos.till_alpha +
                                   2.0 * materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) -
                                   (particles->rho[i] / materials[matId].eos.rho_0))
                              + materials[matId].eos.till_b * particles->rho[i] * particles->e[i]
                                / ((particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                   (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                   (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0))
                                * exp(-materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                      ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)))
                                * (2.0 * materials[matId].eos.till_beta * ((1.0 - (particles->rho[i] / materials[matId].eos.rho_0)) / (particles->rho[i] / materials[matId].eos.rho_0)) *
                                   (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) /
                                   materials[matId].eos.rho_0 + 1.0)
                                / (materials[matId].eos.E_0 * particles->rho[i])
                                * (2.0 * particles->e[i] - particles->p[i] / particles->rho[i]);

                    if (particles->p[i] < 0.0 || (particles->rho[i] / materials[matId].eos.rho_0) < materials[matId].eos.rho_limit) particles->p[i] = 0.0;

                    cs_c_sq = materials[matId].eos.till_a * particles->e[i]
                              + (materials[matId].eos.till_b * particles->e[i]) /
                                ((particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) *
                                 (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0)) *
                                (3.0 * (particles->e[i] / (materials[matId].eos.E_0 * (particles->rho[i] / materials[matId].eos.rho_0) * (particles->rho[i] / materials[matId].eos.rho_0)) + 1.0) - 2.0)
                              + (materials[matId].eos.till_a + 2.0 * materials[matId].eos.till_b * ((particles->rho[i] / materials[matId].eos.rho_0) - 1.0)) / particles->rho[i]
                              + particles->p[i] / (particles->rho[i] * particles->rho[i]) * (materials[matId].eos.till_a * particles->rho[i] + materials[matId].eos.till_b * particles->rho[i] /
                                                                                                                                               ((particles->e[i] / (materials[matId].eos.E_0 *
                                                                                                                                                                    (particles->rho[i] /
                                                                                                                                                                     materials[matId].eos.rho_0) *
                                                                                                                                                                    (particles->rho[i] /
                                                                                                                                                                     materials[matId].eos.rho_0)) +
                                                                                                                                                 1.0) * (particles->e[i] / (materials[matId].eos.E_0 *
                                                                                                                                                                            (particles->rho[i] /
                                                                                                                                                                             materials[matId].eos.rho_0) *
                                                                                                                                                                            (particles->rho[i] /
                                                                                                                                                                             materials[matId].eos.rho_0)) +
                                                                                                                                                         1.0)));

                    y = (particles->e[i] - materials[matId].eos.E_iv) / (materials[matId].eos.E_cv - materials[matId].eos.E_iv);
                    cs_sq = cs_e_sq * (1.0 - y) + cs_c_sq * y;
                }

                if (cs_sq < materials[matId].eos.cs_limit * materials[matId].eos.cs_limit) {
                    particles->cs[i] = materials[matId].eos.cs_limit;
                } else {
                    particles->cs[i] = sqrt(cs_sq);
                }
            }
/*            case EquationOfStates::EOS_TYPE_TILLOTSON: {
                rho = particles->rho[i];
                pressure = particles->p[i];
                eta = rho / materials[matId].eos.rho_0;
                mu = eta - 1.0;
                omega0 = particles->e[i] / (materials[matId].eos.E_0 * eta * eta) + 1.0;
                z = (1.0 - eta) / eta;

                if (eta >= 0.0 || particles->e[i] < materials[matId].eos.E_iv) {
                    if (pressure < 0.0 || eta < materials[matId].eos.rho_limit[matId]) pressure = 0.0;

                    cs_sq = materials[matId].eos.till_a * particles->e[i]
                            + (materials[matId].eos.till_b * particles->e[i]) / (omega0 * omega0) * (3.0 * omega0 - 2.0)
                            + (materials[matId].eos.till_a + 2.0 * materials[matId].eos.till_b * mu) / rho
                            + pressure / (rho * rho) * (materials[matId].eos.till_a * rho + materials[matId].eos.till_b * rho / (omega0 * omega0));
                } else if (particles->e[i] > materials[matId].eos.E_cv) {
                    Gamma_e = materials[matId].eos.till_a
                              + materials[matId].eos.till_b / omega0 * exp(-materials[matId].eos.till_beta * z * z);

                    cs_sq = (Gamma_e + 1.0) * pressure / rho
                            + materials[matId].eos.till_a / rho
                              * exp(-(materials[matId].eos.till_alpha * z + materials[matId].eos.till_beta * z * z))
                              * (1.0 + mu) / (eta * eta)
                              * (materials[matId].eos.till_alpha + 2.0 * materials[matId].eos.till_beta * z - eta)
                            + materials[matId].eos.till_b * rho * particles->e[i]
                              / (omega0 * omega0 * eta * eta)
                              * exp(-materials[matId].eos.till_beta * z * z)
                              * (2.0 * materials[matId].eos.till_beta * z * omega0 / materials[matId].eos.rho_0 + 1.0)
                              / (materials[matId].eos.E_0 * rho)
                              * (2.0 * particles->e[i] - pressure / rho);
                } else {
                    Gamma_e = materials[matId].eos.till_a
                              + materials[matId].eos.till_b / omega0 * exp(-materials[matId].eos.till_beta * z * z);

                    cs_e_sq = (Gamma_e + 1.0) * pressure / rho
                              + materials[matId].eos.till_a / rho
                                * exp(-(materials[matId].eos.till_alpha * z + materials[matId].eos.till_beta * z * z))
                                * (1.0 + mu) / (eta * eta)
                                * (materials[matId].eos.till_alpha + 2.0 * materials[matId].eos.till_beta * z - eta)
                              + materials[matId].eos.till_b * rho * particles->e[i]
                                / (omega0 * omega0 * eta * eta)
                                * exp(-materials[matId].eos.till_beta * z * z)
                                * (2.0 * materials[matId].eos.till_beta * z * omega0 / materials[matId].eos.rho_0 + 1.0)
                                / (materials[matId].eos.E_0 * rho)
                                * (2.0 * particles->e[i] - pressure / rho);

                    if (pressure < 0.0 || eta < materials[matId].eos.rho_limit[matId]) pressure = 0.0;

                    cs_c_sq = materials[matId].eos.till_a * particles->e[i]
                              + (materials[matId].eos.till_b * particles->e[i]) / (omega0 * omega0) * (3.0 * omega0 - 2.0)
                              + (materials[matId].eos.till_a + 2.0 * materials[matId].eos.till_b * mu) / rho
                              + pressure / (rho * rho) * (materials[matId].eos.till_a * rho + materials[matId].eos.till_b * rho / (omega0 * omega0));

                    y = (particles->e[i] - materials[matId].eos.E_iv) / (materials[matId].eos.E_cv - materials[matId].eos.E_iv);
                    cs_sq = cs_e_sq * (1.0 - y) + cs_c_sq * y;
                }

                if (cs_sq < materials[matId].eos.cs_limit * materials[matId].eos.cs_limit) {
                    particles->cs[i] = materials[matId].eos.cs_limit;
                } else {
                    particles->cs[i] = sqrt(cs_sq);
                }
            }*/
                break;
                //case EquationOfStates::EOS_TYPE_ISOTHERMAL_GAS: {
                //    // do nothing since constant
                //} break;
            case EquationOfStates::EOS_TYPE_IDEAL_GAS: {
                particles->cs[i] = cuda::math::sqrt(materials[matId].eos.polytropic_gamma * particles->p[i] /
                                                    particles->rho[i]);
                if (std::isnan(particles->cs[i])) {
                    printf("particles->cs[%i] = %e, gamma = %e, p = %e, rho = %e\n", i, particles->cs[i],
                           materials[matId].eos.polytropic_gamma, particles->p[i], particles->rho[i]);
                    assert(0);
                }
            }
                break;
            case EquationOfStates::EOS_TYPE_LOCALLY_ISOTHERMAL_GAS: {
                real distance = 0.0;
                distance = particles->x[i] * particles->x[i];
#if DIM > 1
                distance += particles->y[i] * particles->y[i];
#if DIM == 3
                distance += particles->z[i] * particles->z[i];
#endif
#endif
                distance = cuda::math::sqrt(distance);
                m_com = 0;
                //TODO: how to calculate cs for EOS_TYPE_ISOTHERMAL_GAS
                //for (j = 0; j < numPointmasses; j++) {
                //    m_com += pointmass.m[j];
                //}
                //double vkep = cuda::math::sqrt(gravConst * m_com/distance);
                //p.cs[i] = vkep * scale_height;
                particles->cs[i] = 0;
            }
                break;
                //default:
                //printf("not implemented!\n");
        }
    }

}

real SPH::Kernel::Launch::initializeSoundSpeed(Particles *particles, Material *materials, int numParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::initializeSoundSpeed, particles, materials, numParticles);
}

real SPH::Kernel::Launch::calculateSoundSpeed(Particles *particles, Material *materials, int numParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateSoundSpeed, particles, materials, numParticles);
}

