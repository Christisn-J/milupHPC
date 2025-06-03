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
    }

    __device__ void murnaghan(Material *materials, Particles *particles, int index){
        particles->p[index]= (materials[particles->materialId[index]].eos.bulk_modulus/materials[particles->materialId[index]].eos.n)*
                (pow(particles->rho[index]/materials[particles->materialId[index]].eos.rho_0, materials[particles->materialId[index]].eos.n) - 1.0);
    }

	__device__ void tillotson(Material *materials, Particles *particles, int index) {
    // Check if the energy is within the compressed region (e <= E_iv)
    if (particles->e[index] <= materials[particles->materialId[index]].eos.till_E_iv) {
        // Compressed region EOS
        particles->p[index] = 
            (materials[particles->materialId[index]].eos.till_a
			+ materials[particles->materialId[index]].eos.till_b / (1 + particles->e[index] / (materials[particles->materialId[index]].eos.till_E_0 * pow((particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0), 2))))
            * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0) * particles->e[index]
            + materials[particles->materialId[index]].eos.till_A * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0)
            + materials[particles->materialId[index]].eos.till_B * pow((particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0),2);
    	}
    // Check if the energy is within the expanded region (e >= E_cv)
    else if (particles->e[index] >= materials[particles->materialId[index]].eos.till_E_cv) {
        // Expanded region EOS
        particles->p[index] =
			materials[particles->materialId[index]].eos.till_a
			* (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0) * particles->e[index]
			+ (materials[particles->materialId[index]].eos.till_b / (1 + particles->e[index] / (materials[particles->materialId[index]].eos.till_E_0 * pow((particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0), 2)))
			* (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0) * particles->e[index]
			+ materials[particles->materialId[index]].eos.till_A * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0)
			* exp(-materials[particles->materialId[index]].eos.till_beta * ((materials[particles->materialId[index]].eos.till_rho_0 / particles->rho[index]) - 1)))
            * exp(-materials[particles->materialId[index]].eos.till_alpha * pow(((materials[particles->materialId[index]].eos.till_rho_0 / particles->rho[index]) - 1), 2));
    	}
    // Partial vaporization region (E_iv < e < E_cv)
    else {
        // Interpolate between the two regions (compressed and expanded)
        particles->p[index] = (1 -
			// interpolation weight
			(particles->e[index] - materials[particles->materialId[index]].eos.till_E_iv) / (materials[particles->materialId[index]].eos.till_E_cv - materials[particles->materialId[index]].eos.till_E_iv)) *
			//Compressed region EOS
            (
                (materials[particles->materialId[index]].eos.till_a
                + materials[particles->materialId[index]].eos.till_b / (1 + particles->e[index] / (materials[particles->materialId[index]].eos.till_E_0 * pow((particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0), 2))))
                * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0) * particles->e[index]
                + materials[particles->materialId[index]].eos.till_A * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0)
                + materials[particles->materialId[index]].eos.till_B * pow((particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0),2)
            ) +
			// interpolation weight
            (particles->e[index] - materials[particles->materialId[index]].eos.till_E_iv) / (materials[particles->materialId[index]].eos.till_E_cv - materials[particles->materialId[index]].eos.till_E_iv) *
			//Expanded region EOS
            (
                materials[particles->materialId[index]].eos.till_a
                * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0) * particles->e[index]
                + (materials[particles->materialId[index]].eos.till_b / (1 + particles->e[index] / (materials[particles->materialId[index]].eos.till_E_0 * pow((particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0), 2)))
                * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0) * particles->e[index]
                + materials[particles->materialId[index]].eos.till_A * (particles->rho[index] / materials[particles->materialId[index]].eos.till_rho_0)
                * exp(-materials[particles->materialId[index]].eos.till_beta * ((materials[particles->materialId[index]].eos.till_rho_0 / particles->rho[index]) - 1)))
                * exp(-materials[particles->materialId[index]].eos.till_alpha * pow(((materials[particles->materialId[index]].eos.till_rho_0 / particles->rho[index]) - 1), 2))
            );
    	}
	}

    __device__ void isothermalGas(Material *materials, Particles *particles, int index) {
        //printf("isothermalGas...\n");
        particles->p[index] = 41255.407 * particles->rho[index];
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
    }

    __device__ void locallyIsothermalGas(Material *materials, Particles *particles, int index) {
        //printf("locallyIsothermalGas...\n");
        particles->p[index] = particles->cs[index] * particles->cs[index] * particles->rho[index];
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
                        ::EOS::tillotson(materials, particles, i);
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

