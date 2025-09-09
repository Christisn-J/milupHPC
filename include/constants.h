//
// Created by Christian Jetter on 09.09.25.
//

#ifndef MILUPHPC_CONSTANTS_H
#define MILUPHPC_CONSTANTS_H

#include "types.h"
#include "parameter.h"
#include "utils/define_checks.h"

#include <string>
#include <limits>

namespace Default {
    constexpr integer integer_value = -1;
    constexpr real real_value = -1.0;
    constexpr const char* string_value = "-";
    constexpr integer numberFiles = 1;
    constexpr integer verbose_lvl = 0;
    constexpr bool loadBalancing = false;
}

namespace Invalid {
    constexpr integer integer_value = -1;
    constexpr real real_value = -1.0;
}

namespace Constants {
    constexpr real
    G = 6.67430e-11;
}

// Template-Struktur
template<typename T>
struct DefaultValue;

template<>
struct DefaultValue<integer> {
    static constexpr integer value() { return Default::integer_value; }
    static std::string str() { return std::to_string(value()); }
};

template<>
struct DefaultValue<real> {
    static constexpr real value() { return Default::real_value; }
    static std::string str() { return std::to_string(value()); }
};

template<>
struct DefaultValue<std::string> {
    static std::string value() { return Default::string_value; }
    static std::string str() { return value(); }
};

// Helper template to get invalid values based on type
template<typename T>
struct InvalidValue;

template<>
struct InvalidValue<integer> {
    static constexpr integer value() { return Invalid::integer_value; }
    static std::string str() { return std::to_string(value()); }
};

template<>
struct InvalidValue<real> {
    static constexpr real value() { return Invalid::real_value; }
    static std::string str() { return std::to_string(value()); }
};

constexpr real
dbl_max = std::numeric_limits<real>::max();
#define DBL_MAX dbl_max;


typedef struct SimulationParameters {
    std::string directory;
    std::string logDirectory;
    int verbosity;
    bool timeKernels;
    int numOutputFiles;
    real timeStep;
    real maxTimeStep;
    real timeEnd;
    bool loadBalancing;
    int loadBalancingInterval;
    int loadBalancingBins;
    std::string inputFile;
    std::string materialConfigFile;
    int outputRank;
    bool performanceLog;
    bool particlesSent2H5;
    int sfcSelection;
    int integratorSelection;
//#if GRAVITY_SIM
    real theta;
    real smoothing;
    int gravityForceVersion;
//#endif
//#if SPH_SIM
    int smoothingKernelSelection;
    int sphFixedRadiusNNVersion;
//#endif
    bool removeParticles;
    int removeParticlesCriterion;
    real removeParticlesDimension;
    int bins;
    bool calculateAngularMomentum;
    bool calculateEnergy;
    bool calculateCenterOfMass;
    real particleMemoryContingent;
    int domainListSize;
} SimulationParameters;

struct To {
    enum Target {
        host, device
    };
    Target t_;

    To(Target t) : t_(t) {}

    operator Target() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

struct Smoothing {
    enum Kernel {
        spiky, cubic_spline, wendlandc2, wendlandc4, wendlandc6
    };
    Kernel t_;

    Smoothing(Kernel t) : t_(t) {}

    operator Smoothing() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

struct Execution {
    enum Location {
        host, device
    };
    Location t_;

    Execution(Location t) : t_(t) {}

    operator Location() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

struct Curve {
    enum Type {
        lebesgue, hilbert
    };
    Type t_;

    Curve(Type t) : t_(t) {}

    operator Type() const { return t_; }
    //friend std::ostream& operator<<(std::ostream& out, const Curve::Type curveType);
private:
    template<typename T>
    operator T() const;
};

struct IntegratorSelection {
    enum Type {
        explicit_euler, predictor_corrector_euler, leapfrog
    };
    Type t_;

    IntegratorSelection(Type t) : t_(t) {}

    operator Type() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

/// implemented equation of states
enum EquationOfStates {
    //EOS_TYPE_ACCRETED = -2, /// special flag for particles that got accreted by a gravitating point mass
    //EOS_TYPE_IGNORE = -1, /// particle is ignored
    EOS_TYPE_POLYTROPIC_GAS = 0, /// polytropic EOS for gas, needs polytropic_K and polytropic_gamma in material.cfg file
    EOS_TYPE_MURNAGHAN = 1, /// Murnaghan EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: rho_0, bulk_modulus, n
    EOS_TYPE_TILLOTSON = 2, /// Tillotson EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: till_rho_0, till_A, till_B, till_E_0, till_E_iv, till_E_cv, till_a, till_b, till_alpha, till_beta; bulk_modulus and shear_modulus are needed to calculate the sound speed and crack growth speed for FRAGMENTATION
    EOS_TYPE_ISOTHERMAL_GAS = 3, /// this is pure molecular hydrogen at 10 K
    //EOS_TYPE_REGOLITH = 4, /// The Bui et al. 2008 soil model
    //EOS_TYPE_JUTZI = 5, /// Tillotson EOS with p-alpha model by Jutzi et al.
    //EOS_TYPE_JUTZI_MURNAGHAN = 6, /// Murnaghan EOS with p-alpha model by Jutzi et al.
    //EOS_TYPE_ANEOS = 7, /// ANEOS (or tabulated EOS in ANEOS format)
    //EOS_TYPE_VISCOUS_REGOLITH = 8, /// describe regolith as a viscous material -> EXPERIMENTAL DO NOT USE
    EOS_TYPE_IDEAL_GAS = 9, /// ideal gas equation, set polytropic_gamma in material.cfg
    //EOS_TYPE_SIRONO = 10, /// Sirono EOS modifed by Geretshauser in 2009/10
    //EOS_TYPE_EPSILON = 11, /// Tillotson EOS with epsilon-alpha model by Wuennemann, Collins et al.
    EOS_TYPE_LOCALLY_ISOTHERMAL_GAS = 12, /// locally isothermal gas: \f$ p = c_s^2 \times \varrho \f$
    //EOS_TYPE_JUTZI_ANEOS = 13/// ANEOS EOS with p-alpha model by Jutzi et al.
};

struct Entry {
    enum Name {
        x,
#if DIM > 1
        y,
#if DIM == 3
        z,
#endif
#endif
        mass
    };
    Name t_;

    Entry(Name t) : t_(t) {}

    operator Name() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

#endif //MILUPHPC_CONSTANTS_H
