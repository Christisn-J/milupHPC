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

#include "material.cuh"
#include "../cuda_utils/cuda_runtime.h"
#include "../parameter.h"
#include "../utils/logger.h"

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

    /**
     * @brief Constructor.
     *
     * @param numMaterials
     * @param ID
     * @param interactions
     * @param alpha
     * @param beta
     */
    MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta);

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
     * @brief Reads a configuration value from the given config setting, or assigns a default invalid value if missing or mismatched.
     *
     * This templated helper function attempts to retrieve a configuration parameter from a libconfig setting.
     * It supports both integer and floating point (`real`) types. The function handles cases where the
     * data type in the config file may not exactly match the expected type:
     *
     * - If the expected type is integer but the config provides a real, it will cast with a warning.
     * - If the expected type is real but the config provides an integer, it will cast safely.
     *
     * If the setting is missing or the type is unsupported, it assigns a predefined invalid value for the type.
     *
     * @tparam T The expected type of the config value (e.g., `integer` or `real`).
     * @param setting Pointer to the parent config setting block.
     * @param name The key name of the parameter to look up.
     * @param outValue Pointer to the variable where the result will be stored.
     * @param id Material ID for logging context.
     */
#include <type_traits>  // For std::is_same

    template<typename T>
    void lookupConfigValueOrDefault(config_setting_t *setting, const char *name, T *outValue, idInteger id) {
        if (!outValue) {
            Logger(ERROR) << "Output pointer is null for material ID " << id;
            return;
        }
        if (!setting) {
            Logger(ERROR) << "Null config setting provided for material ID " << id;
            *outValue = InvalidValue<T>::value;
            return;
        }

        if constexpr(std::is_same<T, integer>::value)
        {
            int tempInt;
            if (config_setting_lookup_int(setting, name, &tempInt)) {
                *outValue = static_cast<integer>(tempInt);
                return;
            } else {
                // Falls Wert als Float vorliegt (optional)
                double tempReal;
                if (config_setting_lookup_float(setting, name, &tempReal)) {
                    *outValue = static_cast<integer>(tempReal);
                    Logger(WARN) << "Casting real to integer for parameter '" << name << "' in material ID " << id;
                    return;
                }
            }
        } else if constexpr(std::is_same<T, real>::value)
        {
            double tempReal;
            if (config_setting_lookup_float(setting, name, &tempReal)) {
                *outValue = static_cast<real>(tempReal);
                return;
            } else {
                // Falls Wert als Int vorliegt (optional)
                int tempInt;
                if (config_setting_lookup_int(setting, name, &tempInt)) {
                    *outValue = static_cast<real>(tempInt);
                    return;
                }
            }
        } else {
            Logger(ERROR) << "lookupConfigValueOrDefault only supports 'integer' or 'real' types.";
            *outValue = InvalidValue<T>::value;
            return;
        }

        Logger(WARN) << "Missing or unsupported type for parameter '" << name << "' in material ID " << id;
        *outValue = InvalidValue<T>::value;
    }
};

#endif //MILUPHPC_MATERIAL_HANDLER_H
