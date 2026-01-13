/**
 * @file material.cuh
 * @brief Material parameters and settings.
 *
 * Material parameters/attributes/properties and settings like:
 *
 * * Equation of state
 * * Artificial viscosity (parameters)
 * * smoothing length
 * * interactions
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo implement missing parameters/variables
 */
#ifndef MILUPHPC_MATERIAL_CUH
#define MILUPHPC_MATERIAL_CUH

#define CHECK_MATERIAL 0

#include "../constants.h"
#include "../cuda_utils/cuda_utilities.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <boost/mpi.hpp>

#if ARTIFICIAL_VISCOSITY
/**
 * @brief Parameters for artificial viscosity.
 *
 * Artificial viscosity is used in SPH simulations to stabilize shocks
 * and prevent particle interpenetration. The standard parameters
 * are alpha and beta.
 *
 * @note TODO: When adding a new parameter to this struct, ensure it is consistently integrated
 *       in all relevant places:
 *       - Compile-time ValueMode constructor (templated)
 *       - Serialization function (for Boost.MPI)
 *       - Material::info() output (both CPU and CUDA versions)
 */
struct ArtificialViscosity {

    // Enable MPI communication (send struct instance directly)
    friend class boost::serialization::access;

    /**
     * @brief Serialization function for Boost.MPI.
     *
     * @tparam Archive Archive type (Boost)
     * @param ar Archive reference
     * @param version Version of serialization
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & alpha;
        ar & beta;
    }

    /// Linear artificial viscosity coefficient (\f$ \alpha \f$)
    real alpha;
    /// Quadratic artificial viscosity coefficient (\f$ \beta \f$)
    real beta;

    /**
    * @brief Default constructor.
    */
    CUDA_CALLABLE_MEMBER ArtificialViscosity() {};

    /**
     * @brief Compile-time constructor using ValueMode.
     *
     * @tparam mode Compile-time ValueMode (DefaultValue or InvalidValue)
     *
     * Initializes all members at compile-time using ValueSelector.
     * Useful for CUDA kernels to avoid runtime branching.
     *
     * @note Optional dummy pointer parameter is used to disambiguate template constructor.
     */
    template<ValueMode mode>
    CUDA_CALLABLE_MEMBER ArtificialViscosity(ValueSelector<mode, real>* = nullptr)
            : alpha(ValueSelector<mode, real>::value()),
              beta(ValueSelector<mode, real>::value()) {};
};
#endif

#if ARTIFICIAL_STRESS
/**
 * @brief Parameters for artificial stress.
 *
 * Artificial stress is used to prevent tensile instability in SPH simulations.
 * Typical parameters include the stress exponent and small numerical coefficients.
 *
 * @note TODO: When adding a new parameter to this struct, ensure it is consistently integrated
 *       in all relevant places:
 *       - Compile-time ValueMode constructor (templated)
 *       - Serialization function (for Boost.MPI)
 *       - Material::info() output (both CPU and CUDA versions)
 */
struct ArtificialStress {

    friend class boost::serialization::access;

    /**
     * @brief Serialization function for Boost.MPI.
     *
     * @tparam Archive Archive type
     * @param ar Archive reference
     * @param version Version number
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & exponent_tensor;
        ar & epsilon_stress;
        ar & mean_particle_distance;
    }

    /// Exponent for the stress tensor
    real exponent_tensor;
    /// Small stress coefficient
    real epsilon_stress;
    /// Reference particle spacing
    real mean_particle_distance;

    /**
    * @brief Default constructor.
    */
    CUDA_CALLABLE_MEMBER ArtificialStress() {};

    /**
     * @brief Compile-time constructor using ValueMode.
     *
     * @tparam mode Compile-time ValueMode (DefaultValue or InvalidValue)
     *
     * Initializes all members at compile-time using ValueSelector.
     * Useful for CUDA kernels to avoid runtime branching.
     *
     * @note Optional dummy pointer parameter is used to disambiguate template constructor.
     */
    template<ValueMode mode>
    CUDA_CALLABLE_MEMBER ArtificialStress(ValueSelector<mode, real>* = nullptr)
        : exponent_tensor(ValueSelector<mode, real>::value()),
          epsilon_stress(ValueSelector<mode, real>::value()),
          mean_particle_distance(ValueSelector<mode, real>::value()) {};
};
#endif

#if PLASTICITY
/**
 * @brief Parameters for plasticity.
 *
 * Plasticity models describe material yielding under stress.
 * The primary parameter is the yield stress.
 *
 * @note TODO: When adding a new parameter to this struct, ensure it is consistently integrated
 *       in all relevant places:
 *       - Compile-time ValueMode constructor (templated)
 *       - Serialization function (for Boost.MPI)
 *       - Material::info() output (both CPU and CUDA versions)
 */
struct Plasticity {

    friend class boost::serialization::access;

    /**
     * @brief Serialization function for Boost.MPI.
     *
     * @tparam Archive Archive type
     * @param ar Archive reference
     * @param version Version number
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & yield_stress;
    }

    /// Yield stress of the material
    real yield_stress;

    /**
    * @brief Default constructor.
    */
    CUDA_CALLABLE_MEMBER Plasticity() {};

    /**
     * @brief Compile-time constructor using ValueMode.
     *
     * @tparam mode Compile-time ValueMode (DefaultValue or InvalidValue)
     *
     * Initializes all members at compile-time using ValueSelector.
     * Useful for CUDA kernels to avoid runtime branching.
     *
     * @note Optional dummy pointer parameter is used to disambiguate template constructor.
     */
    template<ValueMode mode>
    CUDA_CALLABLE_MEMBER Plasticity(ValueSelector<mode, real>* = nullptr)
            : yield_stress(ValueSelector<mode, real>::value()) {};
};
#endif

/**
 * @brief Equation of State (EOS) parameters.
 *
 * This struct contains all parameters required for different EOS types:
 * polytropic, Tillotson, or solid material parameters.
 * Values can be initialized to default or invalid placeholders using ValueMode.
 *
 * @note TODO: When adding a new parameter to this struct, ensure it is consistently integrated
 *       in all relevant places:
 *       - Compile-time ValueMode constructor (templated)
 *       - Serialization function (for Boost.MPI)
 *       - Material::info() output (both CPU and CUDA versions)
 */
struct EqOfSt {

    friend class boost::serialization::access;

    /**
     * @brief Serialization function for Boost.MPI.
     *
     * @tparam Archive Archive type
     * @param ar Archive reference
     * @param version Version number
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
//        ar & yield_stress;
        ar & type;
        ar & rho_0;

        ar & polytropic_K;
        ar & polytropic_gamma;
        ar & bulk_modulus;
        ar & n;
        ar & shear_modulus;
        ar & young_modulus;
        ar & till_A;
        ar & till_B;
        ar & E_0;
        ar & E_iv;
        ar & E_cv;
        ar & till_a;
        ar & till_b;
        ar & till_alpha;
        ar & till_beta;
        ar & rho_limit;
        ar & cs_limit;
    }

//    real yield_stress;

    int type;                ///< EOS type identifier
    real polytropic_K;       ///< Polytropic constant (if polytropic EOS)
    real polytropic_gamma;   ///< Polytropic exponent (if polytropic EOS)
    real rho_0;              ///< Reference density (solids)
    real bulk_modulus;       ///< Bulk modulus (solids)
    real n;                  ///< Material constant for elastic/plastic models
    real shear_modulus;      ///< Shear modulus (solids)
    real young_modulus;      ///< Young's modulus (solids)
    real till_A;             ///< Tillotson EOS coefficient A
    real till_B;             ///< Tillotson EOS coefficient B
    real E_0;                ///< Tillotson reference energy
    real E_iv;               ///< Tillotson incipient vaporization energy
    real E_cv;               ///< Tillotson complete vaporization energy
    real till_a;             ///< Tillotson EOS coefficient a
    real till_b;             ///< Tillotson EOS coefficient b
    real till_alpha;         ///< Tillotson thermal coefficient alpha
    real till_beta;          ///< Tillotson thermal coefficient beta
    real rho_limit;          ///< Minimum allowed density
    real cs_limit;           ///< Minimum allowed sound speed

    /**
    * @brief Default constructor.
    */
    CUDA_CALLABLE_MEMBER EqOfSt() {};

    /**
     * @brief Compile-time constructor using ValueMode.
     *
     * @tparam mode Compile-time ValueMode (DefaultValue or InvalidValue)
     *
     * Initializes all members at compile-time using ValueSelector.
     * Useful for CUDA kernels to avoid runtime branching.
     *
     * @note Optional dummy pointer parameter is used to disambiguate template constructor.
     */
    template<ValueMode mode>
    CUDA_CALLABLE_MEMBER EqOfSt(ValueSelector<mode, real>* = nullptr)
            : type(ValueSelector<mode, integer>::value()),
              rho_0(ValueSelector<mode, real>::value()),
//              yield_stress(ValueSelector<mode, real>::value()),
              polytropic_K(ValueSelector<mode, real>::value()),
              polytropic_gamma(ValueSelector<mode, real>::value()),
              bulk_modulus(ValueSelector<mode, real>::value()),
              n(ValueSelector<mode, real>::value()),
              shear_modulus(ValueSelector<mode, real>::value()),
              young_modulus(ValueSelector<mode, real>::value()),
              till_A(ValueSelector<mode, real>::value()),
              till_B(ValueSelector<mode, real>::value()),
              E_0(ValueSelector<mode, real>::value()),
              E_iv(ValueSelector<mode, real>::value()),
              E_cv(ValueSelector<mode, real>::value()),
              till_a(ValueSelector<mode, real>::value()),
              till_b(ValueSelector<mode, real>::value()),
              till_alpha(ValueSelector<mode, real>::value()),
              till_beta(ValueSelector<mode, real>::value()),
              rho_limit(ValueSelector<mode, real>::value()),
              cs_limit(ValueSelector<mode, real>::value()) {};
};
/**
 * @brief Material parameters class.
 *
 * Stores all physical and numerical parameters associated with a material,
 * including:
 * - Equation of state (EOS)
 * - Artificial viscosity (parameters)
 * - Artificial stress (parameters)
 * - Plasticity parameters
 *
 * Supports MPI serialization and CUDA-compatible operations.
 *
 * Provides two ways to construct Material objects:
 * 1. Default constructor (runtime initialization)
 *    - Members are left uninitialized.
 *
 * 2. Compile-time ValueMode constructor (templated)
 *    - Members are initialized at compile-time using ValueSelector.
 *    - ValueMode can be 'DefaultValue' or 'InvalidValue'.
 *
 * @note TODO: When adding a new parameter or sub-structure to this class,
 *       ensure it is consistently integrated in all relevant places:
 *       - Compile-time ValueMode constructor (templated)
 *       - Serialization function (for Boost.MPI)
 *       - Material::info() output (both CPU and CUDA versions)
 *       - Compile-time ValueMode constructor (if applicable)
 *
 * This design allows CUDA kernels to work with materials without runtime
 * branching and ensures all members are constexpr-initializable if needed.
 */
class Material {
public:
    // Enable MPI communication (allow sending the class instance directly)
    friend class boost::serialization::access;

    /**
     * @brief Serialization function for Boost.MPI.
     *
     * Serializes all members of the material, including optional parameters
     * controlled by compile-time flags.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & ID;
        ar & interactions;
        ar & sml;
#if CHECK_MATERIAL
        ar & dummy_int;
        ar & dummy_real;
#endif
#if ARTIFICIAL_VISCOSITY
        ar & artificialViscosity;
#endif
#if ARTIFICIAL_STRESS
        ar & artificialStress;
#endif
#if PLASTICITY
        ar & plasticity;
#endif
        ar & eos;
    }

    /// Unique material identifier
    integer ID;
    /// Number of interacting neighbors
    integer interactions;
    /// Smoothing length
    real sml;
#if CHECK_MATERIAL
    integer dummy_int;
    real dummy_real;
#endif

#if ARTIFICIAL_VISCOSITY
    /// Parameters for artificial viscosity
    ArtificialViscosity artificialViscosity;
#endif

#if ARTIFICIAL_STRESS
    /// Parameters for artificial stress
    ArtificialStress artificialStress;
#endif

#if PLASTICITY
    /// Parameters for plasticity
    Plasticity plasticity;
#endif

    /// Equation of state (EOS) parameters
    EqOfSt eos;

    /**
     * @brief Default constructor.
     *
     * Initializes a Material object at runtime.
     * Members are left uninitialized unless explicitly set later.
     */
    CUDA_CALLABLE_MEMBER Material();

    /**
     * @brief Compile-time constructor using ValueMode.
     *
     * @tparam mode Compile-time ValueMode (DefaultValue or InvalidValue)
     *
     * Initializes all members at compile-time using ValueSelector.
     * Useful for CUDA kernels to avoid runtime branching.
     *
     * @note Optional dummy pointer parameter is used to disambiguate template constructor.
     */
    template<ValueMode mode>
    CUDA_CALLABLE_MEMBER Material(ValueSelector<mode, integer>* = nullptr)
            : ID(ValueSelector<mode, integer>::value()),
              interactions(ValueSelector<mode, integer>::value()),
              sml(ValueSelector<mode, real>::value()),
#if CHECK_MATERIAL
              dummy_int(ValueSelector<mode, integer>::value()),
              dummy_real(ValueSelector<mode, real>::value()),
#endif
#if ARTIFICIAL_VISCOSITY
              artificialViscosity(static_cast<ValueSelector<mode, real>*>(nullptr)),
#endif
#if ARTIFICIAL_STRESS
              artificialStress(static_cast<ValueSelector<mode, real>*>(nullptr)),
#endif
#if PLASTICITY
              plasticity(static_cast<ValueSelector<mode, real>*>(nullptr)),
#endif
              eos(static_cast<ValueSelector<mode, real>*>(nullptr))
    {}

    /**
     * @brief Destructor.
     *
     * This destructor is intentionally left empty because all member variables
     * of Material are either primitive types or trivially destructible structures.
     */
    CUDA_CALLABLE_MEMBER ~Material();

    /**
     * @brief Print all material parameters.
     *
     * Outputs material properties to the console, useful for debugging or logging.
     * Optional components are printed only if enabled at compile-time.
     */
    CUDA_CALLABLE_MEMBER void info();
};


/* Compared to  milupcuda
 * real density_floor;

   struct eos {
       integer type;
       real shear_modulus;
       real bulk_modulus;
       real yield_stress;

       real cs_limit;

       // ideal gas
       real polytropic_gamma;
       real ideal_gas_rho_0;
       real ideal_gas_p_0;
       real ideal_gas_conv_e_to_T;

       // Tillotson
       real till_rho_0;
       real till_A;
       real till_B;
       real till_E_0;
       real till_E_iv;
       real _E_cv;
       real till_a;
       real till_b;
       real till_alpha;
       real till_beta;
       real rho_limit;

       // ANEOS
       char *table_path;
       integer n_rho;
       integer n_e;
       real aneos_rho_0;
       real aneos_bulk_cs;
       real aneos_e_norm;

       // plasticity
       real cohesion;
       real friction_angle;
       real cohesion_damaged;
       real friction_angle_damaged;
       real melt_energy;


       // fragmentation
       real weibull_k;
       real weibull_m;

   };

   struct porosity {
       real porjutzi_p_elastic;
       real porjutzi_p_transition;
       real porjutzi_p_compacted;
       real porjutzi_alpha_0;
       real porjutzi_alpha_e;
       real porjutzi_alpha_t;
       real porjutzi_n1;
       real porjutzi_n2;
       real cs_porous;
       integer crushcurve_style;
   };

   struct plasticity {
       real yield_stress;
       real cohesion;
       real friction_angle;
       real friction_angle_damaged;
       // ...
   };
*/

/// Material related functions and kernels
namespace MaterialNS {

    /// CUDA kernel functions
    namespace Kernel {

        /**
         * @brief Debug kernel giving information about material(s).
         *
         * > Corresponding wrapper function: ::MaterialNS::Kernel::Launch::info()
         *
         * @param material Material class instance
         */
        __global__ void info(Material *material);

        /// Wrapper functions
        namespace Launch {

            /**
             * @brief Wrapper for ::MaterialNS::Kernel::info().
             */
            void info(Material *material);
        }
    }
}

#endif //MILUPHPC_MATERIAL_CUH
