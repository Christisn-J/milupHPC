#include "../../include/sph/internal_forces.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void SPH::Kernel::internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                                            int *interactions, int numRealParticles) {

    int i, k, inc, j, numInteractions;
    int f, kk;

    real W;
    real tmp;

    real ax;
#if DIM > 1
    real ay;
#if DIM == 3
    real az;
#endif
#endif

    real sml;

    int matId, matIdj;
    real sml1;

    real vxj, vyj, vzj;

    real vr;
    real rr;
    real rhobar;
    real mu;
    real muijmax;
    real smooth;
    real csbar;
    real alpha, beta;

#if BALSARA_SWITCH
    real fi, fj;
    real curli, curlj;
    const real eps_balsara = 1e-4;
#endif

#if ARTIFICIAL_STRESS
    real R_i[DIM][DIM];
    real R_j[DIM][DIM];
    real artf = 0;
    real arts_rij = 0;
    real meanParticleDistance;
    real exponentTensor;
#endif

    int d;
    int dd;
    int e;

    real dr[DIM];
    real dv[DIM];

    real x, vx;
#if DIM > 1
    real y, vy;
#if DIM == 3
    real z, vz;
#endif
#endif

    real drhodt;

#if INTEGRATE_ENERGY
    real dedt;
#endif

    real dvx;
#if DIM > 1
    real dvy;
#if DIM == 3
    real dvz;
#endif
#endif

#if SOLID
    real sigma_i[DIM][DIM], sigma_j[DIM][DIM];
    real S_i[DIM][DIM];
    real edot[DIM][DIM], rdot[DIM][DIM];
    real dSxx;
#if DIM > 1
    real dSxy;
    real dSyy;
#if DIM == 3
    real dSxz;
    real dSyz;
#endif
#endif
    real shear;
    real bulk;
    real young;

    real tensileMax;
#endif // SOLID

#if NAVIER_STOKES
    real eta;
    real zetaij;
#endif

    real vvnablaW;
    real dWdr;
    real dWdrj;
    real dWdx[DIM];
    real Wj;
    real dWdxj[DIM];
    real pij = 0;
    real r;
    real accels[DIM];
    real accelsj[DIM];
    real accelshearj[DIM];

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {

        matId = particles->materialId[i];

        numInteractions = particles->noi[i];

        ax = 0;
#if DIM > 1
        ay = 0;
#if DIM == 3
        az = 0;
#endif
#endif

        alpha = materials[matId].artificialViscosity.alpha; //matAlpha[matId];
        beta = materials[matId].artificialViscosity.beta; //matBeta[matId];
        muijmax = 0;

        sml1 = particles->sml[i];

        drhodt = 0;
#if INTEGRATE_ENERGY
        dedt = 0;
#endif
#if INTEGRATE_SML
        particles->dsmldt[i] = 0.0;
#endif

#if SOLID
        #pragma unroll
        for (d = 0; d < DIM; d++) {
            #pragma unroll
            for (e = 0; e < DIM; e++) {
                // set rotation rate and strain rate tensor to zero
                edot[d][e] = 0.0;
                rdot[d][e] = 0.0;
            }
        }
#endif

        #pragma unroll
        for (d = 0; d < DIM; d++) {
            accels[d] = 0.0;
            accelsj[d] = 0.0;
            accelshearj[d] = 0.0;
        }
        sml = particles->sml[i];

        x = particles->x[i];
#if DIM > 1
        y = particles->y[i];
#if DIM == 3
        z = particles->z[i];
#endif
#endif
        vx = particles->vx[i];
#if DIM > 1
        vy = particles->vy[i];
#if DIM == 3
        vz = particles->vz[i];
#endif
#endif


        //particles->dxdt[i] = 0;
        particles->ax[i] = 0;
#if DIM > 1
        //p.dydt[i] = 0;
        particles->ay[i] = 0;
#if DIM == 3
        //p.dzdt[i] = 0;
        particles->az[i] = 0;
#endif
#endif

        particles->drhodt[i] = 0.0;
#if INTEGRATE_ENERGY
        particles->dedt[i] = 0.0;
#endif
#if INTEGRATE_SML
        particles->dsmldt[i] = 0.0;
#endif

        // if particle has no interactions continue and set all derivs to zero
        // but not the accels (these are handled in the tree for gravity)
        if (numInteractions < 1) {
            // finally continue
            continue;
        }

#if BALSARA_SWITCH
        curli = 0;
        for (d = 0; d < DIM; d++) {
            curli += particles->curlv[i*DIM+d]*particles->curlv[i*DIM+d];
        }
        curli = cuda::math::sqrt(curli);
        fi = cuda::math::abs(particles->divv[i]) / (cuda::math::abs(particles->divv[i])
                + curli + eps_balsara*particles->cs[i]/particles->sml[i]);
#endif

        // THE MAIN SPH LOOP FOR ALL INTERNAL FORCES
        // loop over interaction partners for SPH sums
        for (k = 0; k < numInteractions; k++) {
            //matIdj = EOS_TYPE_IGNORE;
            // the interaction partner
            j = interactions[i * MAX_NUM_INTERACTIONS + k];

            if (i == j) {
                cudaTerminate("i = %i == j = %i\n", i, j);
            }

            matIdj = particles->materialId[j];

            #pragma unroll
            for (d = 0; d < DIM; d++) {
                accelsj[d] = 0.0; // needs to be set to zero for every interaction partner
            }

#if (VARIABLE_SML || INTEGRATE_SML) // || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5 * (particles->sml[i] + particles->sml[j]);
#endif

            vxj = particles->vx[j];
#if DIM > 1
            vyj = particles->vy[j];
#if DIM == 3
            vzj = particles->vz[j];
#endif
#endif
            // relative vector
            dr[0] = x - particles->x[j];
#if DIM > 1
            dr[1] = y - particles->y[j];
#if DIM == 3
            dr[2] = z - particles->z[j];
#endif
#endif
            r = 0;
            #pragma unroll
            for (e = 0; e < DIM; e++) {
                r += dr[e]*dr[e];
                dWdx[e] = 0.0;
#if AVERAGE_KERNELS
                dWdxj[e] = 0.0;
#endif
            }
            W = 0.0;
            dWdr = 0.0;
#if AVERAGE_KERNELS
            Wj = 0.0;
            dWdrj = 0.0;
#endif
            r = cuda::math::sqrt(r);

            // get kernel values for this interaction
#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, particles->sml[i]);
            kernel(&Wj, dWdxj, &dWdrj, dr, particles->sml[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
                dWdxj[e] /= p_rhs.shepard_correction[j];
            }
            dWdr /= p_rhs.shepard_correction[i];
            dWdrj /= p_rhs.shepard_correction[j];

            W = 0.5 * (W + Wj);
            dWdr = 0.5 * (dWdr + dWdrj);
            for (e = 0; e < DIM; e++) {
                dWdx[e] = 0.5 * (dWdx[e] + dWdxj[e]);
            }
# endif // SHEPARD_CORRECTION
#else
            kernel(&W, dWdx, &dWdr, dr, sml);
#if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
            }
            dWdr /= p_rhs.shepard_correction[i];
#endif
#endif

            dv[0] = dvx = vx - vxj;
#if DIM > 1
            dv[1] = dvy = vy - vyj;
#if DIM == 3
            dv[2] = dvz = vz - vzj;
#endif
#endif

            vvnablaW = dvx * dWdx[0];
#if DIM > 1
            vvnablaW += dvy * dWdx[1];
#if DIM == 3
            vvnablaW += dvz * dWdx[2];
#endif
#endif

            rr = 0.0;
            vr = 0.0;
            #pragma unroll
            for (e = 0; e < DIM; e++) {
                rr += dr[e]*dr[e];
                vr += dv[e]*dr[e];
                //printf("vr += %e * %e\n", dv[e], dr[e]);
            }
            //printf("pij: vr = %e\n", vr);

#if SOLID
            // get sigma_i
            SPH::calcStress(particles, sigma_i, i);
            // get sigma_j
            SPH::calcStress(particles, sigma_j, j);

            // calculate edot and rdot
            // edot_ab = 0.5 * (d_b v_a + d_a v_b)
            // rdot_ab = 0.5 * (d_b v_a - d_a v_b)
            tmp = -0.5*particles->mass[j]/particles->rho[j]; // why minus sign? because of dvx = vxi - vxj
            edot[0][0] += tmp*(dvx*dWdx[0] + dvx*dWdx[0]);
#if DIM > 1
            edot[0][1] += tmp*(dvx*dWdx[1] + dvy*dWdx[0]);
            edot[1][1] += tmp*(dvy*dWdx[1] + dvy*dWdx[1]);
            //edot[1][0] += tmp*(dvy*dWdx[0] + dvx*dWdx[1]);

#if DIM == 3
            edot[0][2] += tmp*(dvx*dWdx[2] + dvz*dWdx[0]);
            edot[1][2] += tmp*(dvy*dWdx[2] + dvz*dWdx[1]);
            //edot[2][0] += tmp*(dvz*dWdx[0] + dvx*dWdx[2]);
            //edot[2][1] += tmp*(dvz*dWdx[1] + dvy*dWdx[2]);
            //edot[2][2] += tmp*(dvz*dWdx[2] + dvz*dWdx[2]);
#endif // DIM == 3
#endif // DIM > 1
            rdot[0][0] += tmp*(dvx*dWdx[0] - dvx*dWdx[0]);
#if DIM > 1
            rdot[0][1] += tmp*(dvx*dWdx[1] - dvy*dWdx[0]);
            rdot[1][0] += tmp*(dvy*dWdx[0] - dvx*dWdx[1]);
            rdot[1][1] += tmp*(dvy*dWdx[1] - dvy*dWdx[1]);
#if DIM == 3
            rdot[0][2] += tmp*(dvx*dWdx[2] - dvz*dWdx[0]);
            rdot[1][2] += tmp*(dvy*dWdx[2] - dvz*dWdx[1]);
            rdot[2][0] += tmp*(dvz*dWdx[0] - dvx*dWdx[2]);
            rdot[2][1] += tmp*(dvz*dWdx[1] - dvy*dWdx[2]);
            rdot[2][2] += tmp*(dvz*dWdx[2] - dvz*dWdx[2]);
#endif // DIM == 3
#endif // DIM > 1
#endif // SOLID

            pij = 0.0;

            // artificial viscosity force only if v_ij * r_ij < 0
            if (vr < 0) {
                csbar = 0.5*(particles->cs[i] + particles->cs[j]);
                //if (std::isnan(csbar)) {
                //    printf("csbar = %e, cs[%i] = %e, cs[%i] = %e\n", csbar, i, particles->cs[i], j, particles->cs[j]);
                //}
                smooth = 0.5*(sml1 + particles->sml[j]);

                const real eps_artvisc = 1e-2;
                mu = smooth*vr/(rr + smooth*smooth*eps_artvisc);

                if (mu > muijmax) {
                    muijmax = mu;
                }
                rhobar = 0.5 * (particles->rho[i] + particles->rho[j]);
# if BALSARA_SWITCH
                curlj = 0;
                for (d = 0; d < DIM; d++) {
                    curlj += particles->curlv[j*DIM+d]*particles->curlv[j*DIM+d];
                }
                curlj = cuda::math::sqrt(curlj);
                fj = cuda::math::abs(particles->divv[j]) / (cuda::math::abs(particles->divv[j])
                        + curlj + eps_balsara*particles->cs[j]/particles->sml[j]);
                mu *= (fi+fj)/2.;
# endif
                pij = (beta*mu - alpha*csbar) * mu/rhobar;
                //if (std::isnan(pij) || std::isinf(pij)) {
                //    cudaTerminate("pij = (%e * %e - %e * %e) * (%e/%e), cs[i] = %e, cs[j] = %e\n", beta, mu, alpha, csbar, mu, rhobar,
                //                  particles->cs[i], particles->cs[j]);
                //}
            }

#if NAVIER_STOKES
            eta = (particles->eta[i] + particles->eta[j]) * 0.5 ;
            for (d = 0; d < DIM; d++) {
                accelshearj[d] = 0;
                for (dd = 0; dd < DIM; dd++) {
# if (SPH_EQU_VERSION == 1)
#  if SML_CORRECTION
                    accelshearj[d] += eta * particles->mass[j] * (particles->Tshear[CudaUtils::stressIndex(j,d,dd)]/(particles->sml_omega[j]*particles->rho[j]*particles->rho[j])+ particles->Tshear[stressIndex(i,d,dd)]/(particles->sml_omega[i]*particles->rho[i]*particles->rho[i])) *dWdx[dd];
#  else
                    accelshearj[d] += eta * particles->mass[j] * (particles->Tshear[CudaUtils::stressIndex(j,d,dd)]/(particles->rho[j]*particles->rho[j]) + particles->Tshear[CudaUtils::stressIndex(i,d,dd)]/(particles->rho[i]*particles->rho[i])) *dWdx[dd];
#  endif
# elif (SPH_EQU_VERSION == 2)
#  if SML_CORRECTION
                    accelshearj[d] += eta * particles->mass[j] * (particles->Tshear[CudaUtils::stressIndex(j,d,dd)]+particles->Tshear[CudaUtils::stressIndex(i,d,dd)])/(particles->sml_omega[i]*particles->rho[i]*particles->sml_omega[j]*particles->rho[j]) *dWdx[dd];
#  else
                    accelshearj[d] += eta * particles->mass[j] * (particles->Tshear[CudaUtils::stressIndex(j,d,dd)]+particles->Tshear[CudaUtils::stressIndex(i,d,dd)])/(particles->rho[i]*particles->rho[j]) *dWdx[dd];
#  endif
# endif // SPH_EQU_VERSION
                }
            }
#if KLEY_VISCOSITY //artificial bulk viscosity with f=0.5
            zetaij = 0.0;
            if (vr < 0) { // only for approaching particles
                zetaij = -0.5 * (0.25*(p.h[i] + p.h[j])*(p.h[i]+p.h[j])) * (p.rho[i]+p.rho[j])*0.5 * (p_rhs.divv[i] + p_rhs.divv[j])*0.5;
            }
            for (d = 0; d < DIM; d++) {
# if (SPH_EQU_VERSION == 1)
                accelshearj[d] += zetaij * p.m[j] * (p_rhs.divv[i] + p_rhs.divv[j]) /(p.rho[i]*p.rho[j]) * dWdx[d];
# elif (SPH_EQU_VERSION == 2)
                accelshearj[d] += zetaij * p.m[j] * (p_rhs.divv[i]/(p.rho[i]*p.rho[i]) + p_rhs.divv[j]/(p.rho[j]*p.rho[j])) * dWdx[d];
# endif
            }
#endif // KLEY_VISCOSITY
#endif // NAVIER_STOKES

#if SOLID
#if ARTIFICIAL_STRESS
            meanParticleDistance = materials[matId].artificialStress.mean_particle_distance;
            exponentTensor = materials[matId].artificialStress.exponent_tensor;
            artf = SPH::fixTensileInstability(kernel, particles, i, j, meanParticleDistance );
            artf = cuda::math::pow(artf, exponentTensor);
            // get R_i
            SPH::calcArtificialStress(materials, sigma_i, R_i, matId);
            // get R_j
            SPH::calcArtificialStress(materials, sigma_j, R_j, matIdj);
#endif
            // calculate acceleration for Solids
#pragma unroll
            for (d = 0; d < DIM; d++) {
                accelsj[d] = 0;
#pragma unroll
                for(dd = 0; dd < DIM; dd++){
#if( SPH_EQU_VERSION == 1)
                    accelsj[d] = particles->mass[j] *(sigma_i[d][dd]/(particles->rho[i]*particles->rho[i]) + sigma_j[d][dd]/(particles->rho[j]*particles->rho[j]))*dWdx[dd];
#elif(SPH_EQU_VERSION == 2)
                    accelsj[d] = particles->mass[j] * (sigma_j[d][dd]+sigma_i[d][dd])/(particles->rho[i]*particles->rho[j]) *dWdx[dd];
#else
#error Invalid choice of SPH_EQU_VERSION in parameter.h
#endif // SPH_EQU_VERSION
#if ARTIFICIAL_STRESS
#if (SPH_EQU_VERSION == 1)
                    arts_rij = R_i[d][dd] / (particles->rho[i]*particles->rho[i])
                    + R_j[d][dd] / (particles->rho[j]*particles->rho[j]);
#elif (SPH_EQU_VERSION == 2)
                    arts_rij = ( R_i[d][dd] + R_j[d][dd] ) / (particles->rho[i]*particles->rho[j]);
#endif
                    accels[d] += particles->mass[j] * arts_rij * artf * dWdx[dd];
#endif // ARTIFICIAL_STRESS
                    accels[d] += accelsj[d];
                }
            }
#else // NOT SOLID
            // calculate acceleration for hydro
#if (SPH_EQU_VERSION == 1)
            #pragma unroll
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -particles->mass[j] * (particles->p[i]/(particles->rho[i]*particles->rho[i]) + particles->p[j]/(particles->rho[j]*particles->rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            }
#elif (SPH_EQU_VERSION == 2)
            #pragma unroll
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -particles->mass[j] * ((particles->p[i]+particles->p[j])/(particles->rho[i]*particles->rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            }
#endif // SPH_EQU_VERSION
#endif // SOLID

            //if (std::isnan(accelsj[0])) {
            //    cudaTerminate("accelsj[0] = %e\n", accelsj[0]);
            //}

#if NAVIER_STOKES
            #pragma unroll
            for (d = 0; d < DIM; d++) {
                accels[d] += accelshearj[d];
            }
#endif
            // add artificial viscosity
            accels[0] += particles->mass[j]*(-pij)*dWdx[0];
#if DIM > 1
            accels[1] += particles->mass[j]*(-pij)*dWdx[1];
#if DIM == 3
            accels[2] += particles->mass[j]*(-pij)*dWdx[2];
#endif
#endif
            //if (std::isnan(accels[0])) {
            //    cudaTerminate("accels[0] = %e, mass = %e, pij = %e, dWdx[0] = %e\n", accels[0],
            //                  particles->mass[j], pij, dWdx[0]);
            //}
            // calculate drho/dt
            drhodt += particles->rho[i]/particles->rho[j] * particles->mass[j] * vvnablaW;

#if INTEGRATE_SML
            // minus since vvnablaW is v_i - v_j \nabla W_ij
#if TENSORIAL_CORRECTION
            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    particles->dsmldt[i] -= 1./DIM * particles->sml[i] * particles->mass[j]/particles->rho[j] * dv[d] * dWdx[dd] * particles->tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd];
                }
            }
# else
#  if !SML_CORRECTION
            particles->dsmldt[i] -= 1./DIM * particles->sml[i] * particles->mass[j]/particles->rho[j] * vvnablaW;
#  endif // SML_CORRECTION
#endif
#endif // INTEGRATE_SML

#if INTEGRATE_ENERGY
            if (true) { // !isRelaxationRun) {
                dedt += 0.5 * particles->mass[j] * pij * vvnablaW;
                //if (dedt < 0.) {
                //    printf("dedt (= %e) += 0.5 * %e * %e * %e (= %e)\n", dedt, particles->mass[j], pij, vvnablaW, particles->mass[j] * pij * vvnablaW);
                //}
            }

            // remember, accelsj  are accelerations by particle j, and dv = v_i - v_j
            dedt += 0.5 * accelsj[0] * -dvx;
#  if DIM > 1
            dedt += 0.5 * accelsj[1] * -dvy;
#  endif
#  if DIM > 2
            dedt += 0.5 * accelsj[2] * -dvz;
#  endif
            //if (dedt < 0.) {
            //    printf("dedt (= %e) += (%e + %e + %e) = %e dv (%e, %e, %e)\n", dedt, 0.5 * accelsj[0] * -dvx, 0.5 * accelsj[1] * -dvy, 0.5 * accelsj[2] * -dvz,
            //           0.5 * accelsj[0] * -dvx + 0.5 * accelsj[1] * -dvy + 0.5 * accelsj[2] * -dvz, dvx, dvy, dvz);
            //}

#endif // INTEGRATE ENERGY

        } // neighbors loop end

        ax = accels[0];
#if DIM > 1
        ay = accels[1];
#if DIM == 3
        az = accels[2];
#endif
#endif
        particles->ax[i] = ax;
#if DIM > 1
        particles->ay[i] = ay;
#if DIM == 3
        particles->az[i] = az;
#endif
#endif

        //if (std::isnan(ax)) {
        //    cudaTerminate("ax[%i] = %e, density = %e\n", i, ax, particles->rho[i]);
        //}

        particles->drhodt[i] = drhodt;


#if INTEGRATE_ENERGY
        particles->dedt[i] = dedt;
#endif // INTEGRATE_ENERGY

#if SOLID
        // get S
        // deviatoric stress tensor is symmetric and traceless (only in 3D)
#if DIM == 1
        S_i[0][0] = particles->Sxx[i];
#elif DIM == 2
        S_i[0][0] = particles->Sxx[i];
        S_i[0][1] = S_i[1][0] = particles->Sxy[i];
        S_i[1][1] = particles->Syy[i];
#else // DIM == 3
        S_i[0][0] = particles->Sxx[i];
        S_i[0][1] = S_i[1][0] = particles->Sxy[i];
        S_i[1][1] = particles->Syy[i];
        S_i[0][2] = S_i[2][0] = particles->Sxz[i];
        S_i[1][2] = S_i[2][1] = particles->Syz[i];
        S_i[2][2] = - (particles->Sxx[i] + particles->Syy[i]);
#endif // DIM
        dSxx = 0.0;
#if DIM > 1
        dSxy = 0.0;
        dSyy = 0.0;
#if DIM == 3
        dSxz = 0.0;
        dSyz = 0.0;
#endif // DIM == 3
#endif // DIM > 1
        //calculate dSdt
        shear = materials[matId].eos.shear_modulus;
        bulk = materials[matId].eos.bulk_modulus;
        young = materials[matId].eos.young_modulus;
        dSxx += 2.0 * shear * (edot[0][0] -  edot[0][0] / 3.0);
#if DIM > 1
        dSxy += 2.0 * shear * edot[0][1];
        dSyy += 2.0 * shear * (edot[1][1] - edot[1][1] / 3.0) ;
#if DIM == 3
        dSxz += 2.0 * shear * edot[0][2];
        dSyz += 2.0 * shear * edot[1][2];
#endif // DIM == 3
#endif // DIM > 1
#pragma unroll
        for (d = 0; d < DIM; d++) {
//            // like elastics paper
//            dSxx += S_i[0][d] * rdot[0][d] + rdot[0][d] * S_i[d][0];
//#if DIM > 1
//            dSxy += S_i[0][d] * rdot[1][d] + rdot[0][d] * S_i[d][1];
//            dSyy += S_i[1][d] * rdot[1][d] + rdot[1][d] * S_i[d][1];
//#if DIM == 3
//            dSxz += S_i[0][d] * rdot[2][d] + rdot[0][d] * S_i[d][2];
//            dSyz += S_i[1][d] * rdot[2][d] + rdot[1][d] * S_i[d][2];
//#endif // DIM > 1
//#endif // DIM  == 3

            // like milupcuda paper
            dSxx += S_i[0][d] * rdot[d][0] - rdot[0][d] * S_i[d][0];
#if DIM > 1
            dSxy += S_i[0][d] * rdot[d][1] - rdot[0][d] * S_i[d][1];
            dSyy += S_i[1][d] * rdot[d][1] - rdot[1][d] * S_i[d][1];
#if DIM  == 3
            dSxz += S_i[0][d] * rdot[d][2] - rdot[0][d] * S_i[d][2];
            dSyz += S_i[1][d] * rdot[d][2] - rdot[1][d] * S_i[d][2];
#endif // DIM > 1
#endif // DIM  == 3

        }

// remember dSdt
        particles->dSdtxx[i] = dSxx;
//        if(i == 1){
//            printf("INTERNALFORCES: Sxx: %e, dSxx: %e /n",particles->Sxx[i], particles->dSdtxx[i] );
//        }
#if DIM > 1
        particles->dSdtxy[i] = dSxy;
        particles->dSdtyy[i] = dSyy;
#if DIM == 3
        particles->dSdtxz[i] = dSxz;
        particles->dSdtyz[i] = dSyz;
#endif // DIM == 3
#endif // DIM > 1
        tensileMax = 0.0;
        tensileMax = CudaUtils::calculateMaxEigenvalue(sigma_i);
        particles->localStrain[i] = tensileMax/young;
#endif // SOLID


    } // particle loop end

}


real SPH::Kernel::Launch::internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                    int *interactions, int numRealParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::internalForces, kernel, materials, tree, particles, interactions,
                        numRealParticles);
}


