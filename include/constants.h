/**
 * @file constants.h
 * @brief Default, invalid, and physical constants with type-safe selectors
 *
 * This header provides:
 * - Default and invalid sentinel values for integer, real, and string types
 * - Physical constants (e.g., gravitational constant)
 * - Type-safe templates to select default or invalid values
 * - Enumerations and structs for execution targets, smoothing kernels, integrators, and EOS types
 * - Particle entry field definitions based on the simulation dimension
 *
 * The goal is to centralize all constants and default values in one place, with compile-time
 * type safety and descriptive string conversion where appropriate.
 *
 * @author Christian Jetter
 * @date 09.09.25
 * @bug No known bugs
 */

#ifndef MILUPHPC_CONSTANTS_H
#define MILUPHPC_CONSTANTS_H

#include "types.h"
#include "parameter.h"
#include "utils/define_checks.h"

#include <limits>

/**
 * @brief Enumeration for selecting default or invalid values.
 */
enum class ValueMode {
    DefaultValue, ///< Use default values
    InvalidValue  ///< Use invalid sentinel values
};

/// @brief Namespace holding default values for various types
namespace Default {
    constexpr integer integer_value = 0;   ///< Default integer value
    constexpr real real_value = 0.0;       ///< Default real value
    constexpr const char* string_value = "-"; ///< Default string value
    constexpr integer numberFiles = 1;     ///< Default number of files
    constexpr integer verbose_lvl = 0;     ///< Default verbosity level
    constexpr bool loadBalancing = false;  ///< Default load balancing flag
}

/// @brief Namespace holding physical constants
namespace Constants {
    namespace physics{
        constexpr real G = 6.67430e-11; ///< Gravitational constant
    }
    namespace numerics{
        constexpr real dbl_max = std::numeric_limits<real>::max();
        constexpr real dbl_min = std::numeric_limits<real>::min();
    }
}

#define DBL_MAX Constants::numerics::dbl_max
#define DBL_MIN Constants::numerics::dbl_min

/// @brief Namespace holding invalid values for various types
namespace Invalid {
    constexpr integer integer_value = std::numeric_limits<integer>::min(); ///< Minimum integer as invalid value
    constexpr real real_value = -std::numeric_limits<real>::infinity();    ///< Negative infinity as invalid real
}

/**
 * @brief Template to select a value based on ValueMode.
 *
 * @tparam MODE Either DefaultValue or InvalidValue
 * @tparam T Type of the value
 */
template<ValueMode MODE, typename T>
struct ValueSelector;

template<typename T>
struct DefaultValue;

template<typename T>
struct InvalidValue;

/**
 * @brief Specialization of ValueSelector for default values
 */
template<typename T>
struct ValueSelector<ValueMode::DefaultValue, T> {
    static constexpr T value() { return DefaultValue<T>::value(); }
};

/**
 * @brief Specialization of ValueSelector for invalid values
 */
template<typename T>
struct ValueSelector<ValueMode::InvalidValue, T> {
    static constexpr T value() { return InvalidValue<T>::value(); }
};

// Default value specializations
template<>
struct DefaultValue<integer> {
    static constexpr integer value() { return Default::integer_value; }
    static std::string str() { return std::to_string(value()); }
};

template<>
struct DefaultValue<real> {
    static constexpr real value() { return ::Default::real_value; }
    static std::string str() { return std::to_string(value()); }
};

template<>
struct DefaultValue<std::string> {
    static std::string value() { return ::Default::string_value; }
    static std::string str() { return value(); }
};

// Invalid value specializations
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

/**
 * @brief Target location for execution or memory operations
 */
struct To {
    enum Target {
        host,   ///< CPU
        device  ///< GPU
    };
    Target t_;

    To(Target t) : t_(t) {}
    operator Target() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

/**
 * @brief Smoothing kernel types
 */
struct Smoothing {
    enum Kernel {
        spiky,
        cubic_spline,
        wendlandc2,
        wendlandc4,
        wendlandc6
    };
    Kernel t_;

    Smoothing(Kernel t) : t_(t) {}
    operator Smoothing() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

/**
 * @brief Execution location for computations
 */
struct Execution {
    enum Location {
        host,   ///< CPU
        device  ///< GPU
    };
    Location t_;

    Execution(Location t) : t_(t) {}
    operator Location() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

/**
 * @brief Space-filling curve types
 */
struct Curve {
    enum Type {
        lebesgue,
        hilbert
    };
    Type t_;

    Curve(Type t) : t_(t) {}
    operator Type() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

/**
 * @brief Integrator selection types for time integration
 */
struct IntegratorSelection {
    enum Type {
        explicit_euler,
        predictor_corrector_euler,
        leapfrog
    };
    Type t_;

    IntegratorSelection(Type t) : t_(t) {}
    operator Type() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

/**
 * @brief Equation of State (EOS) types
 */
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


/**
 * @brief Particle entry fields
 */
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

    explicit Entry(Name t) : t_(t) {}
    operator Name() const { return t_; }

private:
    template<typename T>
    operator T() const;
};

#endif // MILUPHPC_CONSTANTS_H
