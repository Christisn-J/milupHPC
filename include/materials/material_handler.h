/**
 * @file material_handler.h
 * @brief Handler for material parameters and settings.
 *
 * Handler for material parameters/attributes/properties and settings like:
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
#ifndef MILUPHPC_MATERIAL_HANDLER_H
#define MILUPHPC_MATERIAL_HANDLER_H

#include "../constants.h"
#include "../cuda_utils/cuda_runtime.h"
#include "../utils/logger.h"
#include "material.cuh"

#include <fstream>
#include <libconfig.h>

/**
 * @brief Read material config files.
 */
class LibConfigReader {
public:
    config_t config;
    config_setting_t *materials;

    /**
     * Load/read config file.
     *
     * @param configFile provided config file/path
     * @return number of materials within provided config file
     */
    int loadConfigFromFile(const char *configFile);
};

/**
 * @brief Material class handler.
 *
 * * handling host and device instances
 * * initializing values using `LibConfigReader`
 * * copying instances/values between MPI processes and/or device and host
 */
class MaterialHandler {

public:
    /// number of materials or rather material instances
    integer numMaterials;
    /// host instance of material class
    Material *h_materials;
    /// device instance of material class
    Material *d_materials;

    /**
     * @brief Constructor.
     *
     * @param numMaterials
     */
    MaterialHandler(integer numMaterials);

    /**
     * @brief Constructor from config file.
     *
     * @param material_cfg Config file name/path
     */
    MaterialHandler(const char *material_cfg);

//    /**
//     * @brief Constructor.
//     *
//     * @param numMaterials
//     * @param ID
//     * @param interactions
//     * @param alpha
//     * @param beta
//     */
//    MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta);

    /**
     * @brief Destructor.
     */
    ~MaterialHandler();

    /**
     * Copy material instance(s) from host to device or vice-versa.
     *
     * @param target target: host or device
     * @param index material instance index to be copied, if `-1` copy all instances
     */
    void copy(To::Target target, integer index = -1);

    /**
     * Communicate material instances between MPI processes and in addition
     * from and/or to the device(s).
     *
     * @warning it is not possible to send it from device to device via CUDA-aware MPI,
     * since serialize functionality not usable on device
     *
     * @param from MPI process source
     * @param to MPI process target
     * @param fromDevice flag whether start from device
     * @param toDevice flag whether start from device
     */
    void communicate(integer from, integer to, bool fromDevice = false, bool toDevice = true);

    /**
     * Broadcast material instances to all MPI processes from a root
     *
     * @param root root to broadcast from (default: MPI process 0)
     * @param fromDevice flag whether start from device
     * @param toDevice flag whether start from device
     */
    void broadcast(integer root = 0, bool fromDevice = false, bool toDevice = true);

private:
    /**
     * @brief Lookup a configuration value of type T from a config setting.
     *
     * This function retrieves a configuration value from a given setting and stores it in `outValue`.
     * If the setting or the requested value is missing, or if the type is unsupported, the function
     * logs an error and exits without modifying `outValue`.
     *
     * @tparam T The type of the configuration value to retrieve. Supported types are `integer` and `real`.
     * @param setting Pointer to the configuration setting.
     * @param name Name of the parameter to look up.
     * @param outValue Pointer to the output variable where the value will be stored.
     * @param id Material or entity ID used for logging purposes.
     */
#include <type_traits>

    enum class LookupMode { Required, Optional };

    template<typename T>
    void lookupValue(config_setting_t *setting, const char *name, T *outValue, idInteger id, LookupMode mode) {
        if (!outValue) {
            Logger(ERROR) << "Output pointer is null for material ID " << id;
            return;
        }
        if (!setting) {
            if (mode == LookupMode::Required) {
                Logger(ERROR) << "Null config setting for required parameter '" << name
                              << "' in material ID " << id;
            } else {
                Logger(WARN) << "Null config setting for optional parameter '" << name
                             << "' in material ID " << id;
                *outValue = InvalidValue<T>::value();
            }
            return;
        }

        bool found = false;

        if constexpr(std::is_same<T, integer>::value) {
            int tempInt;
            if (config_setting_lookup_int(setting, name, &tempInt)) {
                *outValue = static_cast<integer>(tempInt);
                found = true;
            } else {
                double tempReal;
                if (config_setting_lookup_float(setting, name, &tempReal)) {
                    *outValue = static_cast<integer>(tempReal);
                    Logger(mode == LookupMode::Required ? ERROR : WARN)
                            << "Casting real to integer for parameter '" << name << "' in material ID " << id;
                    found = true;
                }
            }
        } else if constexpr(std::is_same<T, real>::value) {
            double tempReal;
            if (config_setting_lookup_float(setting, name, &tempReal)) {
                *outValue = static_cast<real>(tempReal);
                found = true;
            } else {
                int tempInt;
                if (config_setting_lookup_int(setting, name, &tempInt)) {
                    *outValue = static_cast<real>(tempInt);
                    found = true;
                }
            }
        } else {
            Logger(ERROR) << "lookup only supports 'integer' or 'real' types.";
            *outValue = InvalidValue<T>::value();
            return;
        }

        if (!found) {
            if (mode == LookupMode::Required) {
                Logger(ERROR) << "Missing or unsupported type for required parameter '"
                              << name << "' in material ID " << id;
            } else {
                Logger(WARN) << "Missing or unsupported type for optional parameter '"
                             << name << "' in material ID " << id;
                *outValue = InvalidValue<T>::value();
            }
        }
    }
};


#endif //MILUPHPC_MATERIAL_HANDLER_H
