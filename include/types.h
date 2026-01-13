
/**
 * @file types.h
 * @brief Defines basic types and the main simulation parameters structure.
 *
 * This file provides type aliases and the `SimulationParameters` struct
 * which holds the global configuration for a simulation.
 *
 * Type definitions:
 * - `real`: Floating point type used throughout the program. Can be `float` or `double` depending on precision.
 * - `integer`: Standard integer type.
 * - `keyType`: Unsigned integer used for tree keys. Influences the maximal tree depth:
 *     \f[ \text{maximal tree depth} = \frac{\text{sizeof(keyType)} - (\text{sizeof(keyType)} \% \text{DIM})}{\text{DIM}} \f]
 * - `idInteger`: Integer type for IDs.
 *
 * @author Christian Jetter
 * @date 09.09.25
 * @bug No known bugs
 */

#ifndef MILUPHPC_TYPES_H
#define MILUPHPC_TYPES_H

#include <string>

#ifdef SINGLE_PRECISION
typedef float real; ///< Single precision floating point
#else
typedef double real; ///< Double precision floating point
#endif

typedef int integer;           ///< Standard integer type
typedef unsigned long keyType; ///< Type for tree keys
typedef int idInteger;         ///< Type for IDs

/**
 * @brief Structure containing all simulation configuration parameters.
 *
 * This struct stores paths, numerical settings, physics options,
 * output settings, and other configuration details for a simulation run.
 */
typedef struct SimulationParameters {
    std::string directory;              ///< Directory for simulation output
    std::string logDirectory;           ///< Directory for log files
    int verbosity;                      ///< Verbosity level for logging
    bool timeKernels;                   ///< Measure kernel execution times
    int numOutputFiles;                 ///< Number of output files to write
    real timeStep;                      ///< Simulation time step
    real maxTimeStep;                   ///< Maximum allowed time step
    real timeEnd;                       ///< End time of simulation
    bool loadBalancing;                 ///< Enable dynamic load balancing
    int loadBalancingInterval;          ///< Interval (in steps) for load balancing
    int loadBalancingBins;              ///< Number of bins used for load balancing
    std::string inputFile;              ///< Input file path
    std::string materialConfigFile;     ///< Material configuration file
    int outputRank;                      ///< MPI rank responsible for output
    bool performanceLog;                ///< Enable performance logging
    bool particlesSent2H5;              ///< Store particles to HDF5 format
    int sfcSelection;                   ///< Space-filling curve selection
    int integratorSelection;            ///< Integration method selection
//#if GRAVITY_SIM
    real theta;                         ///< Barnes-Hut opening angle for gravity
    real smoothing;                     ///< Gravitational softening length
    int gravityForceVersion;            ///< Gravity force calculation version
//#endif
//#if SPH_SIM
    int smoothingKernelSelection;       ///< Selection of SPH smoothing kernel
    int sphFixedRadiusNNVersion;        ///< SPH fixed-radius nearest neighbor version
//#endif
    bool removeParticles;               ///< Enable particle removal
    int removeParticlesCriterion;       ///< Criterion for particle removal
    real removeParticlesDimension;      ///< Dimension used for particle removal
    int bins;                            ///< Number of bins (domain decomposition)
    bool calculateAngularMomentum;      ///< Compute angular momentum
    bool calculateEnergy;               ///< Compute total energy
    bool calculateCenterOfMass;         ///< Compute center of mass
    real particleMemoryContingent;      ///< Memory contingency per particle
    int domainListSize;                 ///< Maximum size of the domain list
} SimulationParameters;

#endif //MILUPHPC_TYPES_H
