//
// Created by Christian Jetter on 09.09.25.
// Purpose: Compile-time checks for configuration defines in parameter.h
//

#ifndef MILUPHPC_DEFINE_CHECKS_H
#define MILUPHPC_DEFINE_CHECKS_H

#include "../parameter.h"

// ------------------------------
// DIMENSION CHECK
// ------------------------------
#if DIM < 1 || DIM > 3
#error "DIM must be between 1 and 3"
#endif

static_assert(DIM >= 1 && DIM <= 3, "DIM must be between 1 and 3");

// ------------------------------
// SAFETY LEVEL CHECK
// ------------------------------
#if SAFETY_LEVEL < 0 || SAFETY_LEVEL > 3
#error "SAFETY_LEVEL must be between 0 and 3"
#endif


// ------------------------------
// BOOLEAN FLAGS CHECK
// ------------------------------
// Macro to check flags that must be 0 or 1
#define CHECK_BOOL_FLAG(flag) static_assert((flag) == 0 || (flag) == 1, #flag " must be 0 or 1")

CHECK_BOOL_FLAG(DEBUGGING);
CHECK_BOOL_FLAG(LOGCOLOR);
CHECK_BOOL_FLAG(SI_UNITS);
CHECK_BOOL_FLAG(CUBIC_DOMAINS);
CHECK_BOOL_FLAG(GRAVITY_SIM);
CHECK_BOOL_FLAG(SPH_SIM);
CHECK_BOOL_FLAG(INTEGRATE_ENERGY);
CHECK_BOOL_FLAG(INTEGRATE_DENSITY);
CHECK_BOOL_FLAG(INTEGRATE_SML);
CHECK_BOOL_FLAG(DECOUPLE_SML);
CHECK_BOOL_FLAG(VARIABLE_SML);
CHECK_BOOL_FLAG(SML_CORRECTION);
CHECK_BOOL_FLAG(ARTIFICIAL_VISCOSITY);
CHECK_BOOL_FLAG(BALSARA_SWITCH);
CHECK_BOOL_FLAG(AVERAGE_KERNELS);
CHECK_BOOL_FLAG(DEAL_WITH_TOO_MANY_INTERACTIONS);
CHECK_BOOL_FLAG(SHEPARD_CORRECTION);
CHECK_BOOL_FLAG(SOLID);
CHECK_BOOL_FLAG(NAVIER_STOKES);
CHECK_BOOL_FLAG(ARTIFICIAL_STRESS);
CHECK_BOOL_FLAG(POROSITY);
CHECK_BOOL_FLAG(ZERO_CONSISTENCY);
CHECK_BOOL_FLAG(LINEAR_CONSISTENCY);
CHECK_BOOL_FLAG(FRAGMENTATION);
CHECK_BOOL_FLAG(PALPHA_POROSITY);
CHECK_BOOL_FLAG(PLASTICITY);
CHECK_BOOL_FLAG(KLEY_VISCOSITY);

// ------------------------------
// SPH & GRAVITY SIMULATION VALIDATION
// ------------------------------

// SPH simulation requires density integration
//#if SPH_SIM && !INTEGRATE_DENSITY
//#error "SPH_SIM requires INTEGRATE_DENSITY to be enabled"
//#endif

// Variable smoothing length requires integration of smoothing length
#if VARIABLE_SML && !INTEGRATE_SML
#error "VARIABLE_SML requires INTEGRATE_SML to be enabled"
#endif

// Decoupled smoothing length also requires integration
#if DECOUPLE_SML && !INTEGRATE_SML
#error "DECOUPLE_SML requires INTEGRATE_SML to be enabled"
#endif

// Artificial viscosity only makes sense with SPH
//#if ARTIFICIAL_VISCOSITY && !SPH_SIM
//#error "ARTIFICIAL_VISCOSITY is only supported with SPH_SIM"
//#endif

// P-alpha porosity model requires porosity to be enabled
#if PALPHA_POROSITY && !POROSITY
#error "PALPHA_POROSITY requires POROSITY"
#endif

// ------------------------------
// NUMERICAL PARAMETER CHECKS
// ------------------------------

// Domain list size must be valid
#if DOMAIN_LIST_SIZE < 1
#error "DOMAIN_LIST_SIZE must be at least 1"
#endif

// Courant factor must be in (0, 1]
static_assert(COURANT_FACT > 0 && COURANT_FACT <= 1, "COURANT_FACT must be in the range (0, 1]");

// Forces factor must be in (0, 1]
static_assert(FORCES_FACT > 0 && FORCES_FACT <= 1, "FORCES_FACT must be in the range (0, 1]");


// ------------------------------
// EXPERIMENTAL / DEPRECATED FLAGS
// ------------------------------

// Warning for experimental kernel averaging
#if AVERAGE_KERNELS
#warning "AVERAGE_KERNELS is experimental and may not be fully supported"
#endif

#endif // MILUPHPC_DEFINE_CHECKS_H


