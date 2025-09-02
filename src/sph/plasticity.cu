#if PLASTICITY
__device__ void SPH::calcPlasticity(Material *materials, int matId, Particles *particles, real sigma[DIM][DIM], int index) {
    double mises_f = 1.0;
    double J2 = 0.0;
    double y;

    double trace = 0.0;

#if VON_MISES_PLASTICITY
    #if DIM == 1
        trace = sigma[0][0];
        J2 = 0.5 * ((sigma[0][0] - trace) * (sigma[0][0] - trace));
    #elif DIM == 2
        trace = 0.5 * (sigma[0][0] + sigma[1][1]);
        J2 = 0.5 * (
            (sigma[0][0] - trace) * (sigma[0][0] - trace) +
            (sigma[1][1] - trace) * (sigma[1][1] - trace) +
            2.0 * sigma[0][1] * sigma[0][1]
        );
    #elif DIM == 3
        trace = (sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;
        J2 = 0.5 * (
            (sigma[0][0] - trace) * (sigma[0][0] - trace) +
            (sigma[1][1] - trace) * (sigma[1][1] - trace) +
            (sigma[2][2] - trace) * (sigma[2][2] - trace) +
            2.0 * (
                sigma[0][1] * sigma[0][1] +
                sigma[0][2] * sigma[0][2] +
                sigma[1][2] * sigma[1][2]
            )
        );
    #endif
#else
    #if DIM == 1
        J2 = 0.5 * (sigma[0][0] * sigma[0][0]);
    #elif DIM == 2
        J2 = 0.5 * (
            sigma[0][0] * sigma[0][0] +
            sigma[1][1] * sigma[1][1] +
            2.0 * sigma[0][1] * sigma[0][1]
        );
    #elif DIM == 3
        J2 = 0.5 * (
            sigma[0][0] * sigma[0][0] +
            sigma[1][1] * sigma[1][1] +
            sigma[2][2] * sigma[2][2] +
            2.0 * (
                sigma[0][1] * sigma[0][1] +
                sigma[0][2] * sigma[0][2] +
                sigma[1][2] * sigma[1][2]
            )
        );
    #else
        #error "DIM must be 1, 2, or 3"
    #endif
#endif

    y = materials[matId].eos.yieldStress;
    if (J2 > 0.0) {
        mises_f = (y * y) / (3.0 * J2);
    }
    if (mises_f > 1.0) {
        mises_f = 1.0;
    }

#if VON_MISES_PLASTICITY
    #if DIM >= 1
        sigma[0][0] = mises_f * (sigma[0][0] - trace) + trace;
    #endif
    #if DIM >= 2
        sigma[1][1] = mises_f * (sigma[1][1] - trace) + trace;
        sigma[0][1] = sigma[1][0] = mises_f * sigma[0][1];
    #endif
    #if DIM >= 3
        sigma[2][2] = mises_f * (sigma[2][2] - trace) + trace;
        sigma[0][2] = sigma[2][0] = mises_f * sigma[0][2];
        sigma[1][2] = sigma[2][1] = mises_f * sigma[1][2];
    #endif
#else
    #if DIM >= 1
    // keine Schubspannung, nichts zu skalieren
    #endif

    #if DIM >= 2
        sigma[0][1] = sigma[1][0] *= mises_f;
    #endif

    #if DIM >= 3
        sigma[0][2] = sigma[2][0] *= mises_f;
        sigma[1][2] = sigma[2][1] *= mises_f;
    #endif
#endif
}
#endif
