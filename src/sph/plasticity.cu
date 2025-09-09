#if PLASTICITY

__device__ void SPH::applyPlasticity(Material *materials, int matId, Particles *particles, int index) {
    double mises_f = 1.0;
    double J2 = 0.0;
    double y_0 = materials[matId].eos.yieldStress;

// --- Access deviatoric stress tensor components ---
#if DIM >= 1
    double Sxx = particles->Sxx[index];
#endif

#if DIM >= 2
    double Sxy = particles->Sxy[index];
    double Syy = particles->Syy[index];
#endif

#if DIM >= 3
    double Sxz = particles->Sxz[index];
    double Syz = particles->Syz[index];
    // Szz is not stored explicitly — compute it from the traceless condition
    double Szz = -(Sxx + Syy);
#endif

// --- Compute second invariant J2 of the deviatoric stress tensor ---
#if DIM >= 1
    J2 += 0.5 * (Sxx * Sxx);
#endif

#if DIM >= 2
    J2 += 0.5 * (Syy * Syy + 2.0 * Sxy * Sxy); // shear terms count twice
#endif

#if DIM >= 3
    J2 += 0.5 * (Szz * Szz + 2.0 * Sxz * Sxz + 2.0 * Syz * Syz);
#endif

// --- Compute yield scaling factor fY ---
if (J2 > 0.0) {
    mises_f = fmin((y_0 * y_0) / (3.0 * J2), 1.0);
}

// --- Apply scaling to deviatoric stress components ---
#if DIM >= 1
    particles->Sxx[index] *= mises_f;
#endif

#if DIM >= 2
    particles->Sxy[index] *= mises_f;
    particles->Syy[index] *= mises_f;
#endif

#if DIM >= 3
    particles->Sxz[index] *= mises_f;
    particles->Syz[index] *= mises_f;
    // No need to scale Szz — it's computed, not stored
#endif

}
//
//__device__ void SPH::calcPlasticity(Material *materials, int matId, Particles *particles, real sigma[DIM][DIM], int index) {
//    double mises_f = 1.0;
//    double J2 = 0.0;
//    double y;
//
//    double trace = 0.0;
//
//#if VON_MISES_PLASTICITY
//    #if DIM == 1
//        trace = sigma[0][0];
//        J2 = 0.5 * ((sigma[0][0] - trace) * (sigma[0][0] - trace));
//    #elif DIM == 2
//        trace = 0.5 * (sigma[0][0] + sigma[1][1]);
//        J2 = 0.5 * (
//            (sigma[0][0] - trace) * (sigma[0][0] - trace) +
//            (sigma[1][1] - trace) * (sigma[1][1] - trace) +
//            2.0 * sigma[0][1] * sigma[0][1]
//        );
//    #elif DIM == 3
//        trace = (sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;
//        J2 = 0.5 * (
//            (sigma[0][0] - trace) * (sigma[0][0] - trace) +
//            (sigma[1][1] - trace) * (sigma[1][1] - trace) +
//            (sigma[2][2] - trace) * (sigma[2][2] - trace) +
//            2.0 * (
//                sigma[0][1] * sigma[0][1] +
//                sigma[0][2] * sigma[0][2] +
//                sigma[1][2] * sigma[1][2]
//            )
//        );
//    #endif
//#endif
//
//    y = materials[matId].eos.yieldStress;
//    if (J2 > 0.0) {
//        mises_f = (y * y) / (3.0 * J2);
//    }
//    if (mises_f > 1.0) {
//        mises_f = 1.0;
//    }
//
//#if VON_MISES_PLASTICITY
//    #if DIM >= 1
//        sigma[0][0] = mises_f * (sigma[0][0] - trace) + trace;
//    #endif
//    #if DIM >= 2
//        sigma[1][1] = mises_f * (sigma[1][1] - trace) + trace;
//        sigma[0][1] = sigma[1][0] = mises_f * sigma[0][1];
//    #endif
//    #if DIM >= 3
//        sigma[2][2] = mises_f * (sigma[2][2] - trace) + trace;
//        sigma[0][2] = sigma[2][0] = mises_f * sigma[0][2];
//        sigma[1][2] = sigma[2][1] = mises_f * sigma[1][2];
//    #endif
//}
#endif
