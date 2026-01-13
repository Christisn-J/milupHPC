/**
 * @file compile_checks.h
 * @brief Runtime validation of configuration values with optional fallback and termination logic
 *
 * This header provides template functions and utility functions to validate
 * runtime configuration values, including:
 * - Ensuring numerical parameters are within specified bounds
 * - Checking boolean parameters
 * - Validating the availability of files and directories
 *
 * If a value is invalid, the functions can either:
 * - Log a warning and return a fallback value
 * - Log an error and terminate the program (optionally via MPI_Finalize)
 *
 * @author Christian Jetter
 * @date 09.09.25
 * @bug No known bugs
 */

#ifndef MILUPHPC_COMPILE_CHECKS_H
#define MILUPHPC_COMPILE_CHECKS_H

#include "logger.h"
#include "../constants.h"

#include <string>
#include <fstream>
#include <cstdlib>
#include <mpi.h>
#include <boost/filesystem.hpp>  // For directory checks

// ------------------------------
// VALUE CONSTRAINT CHECKS
// ------------------------------

/**
 * @brief Ensures a value is at least `min`. Returns `fallback` if invalid.
 */
template<typename T>
T checkMinValue(T value, T min, T fallback, const std::string& name,
                const std::string& source = "value", bool terminate = false)
{
    Logger(CHECK) << "Checking minimal value " << name;
    if (value < min) {
        Logger(WARN) << "Invalid " << name << " from " << source << ": " << value
                     << ". Resetting to: " << fallback;
        if (terminate) {
            Logger(ERROR) << "Terminating due to invalid parameter: " << name;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
        return fallback;
    }
    Logger(CHECK) << "Parameter " << name << " checked, set at " << value;
    return value;
}

/**
 * @brief Ensures a value is at most `max`. Returns `fallback` if invalid.
 */
template<typename T>
T checkMaxValue(T value, T max, T fallback, const std::string& name,
                const std::string& source = "value", bool terminate = false)
{
    Logger(CHECK) << "Checking maximal value " << name;
    if (value > max) {
        Logger(WARN) << "Invalid " << name << " from " << source << ": " << value
                     << " exceeds maximum " << max << ". Resetting to: " << fallback;
        if (terminate) {
            Logger(ERROR) << "Terminating due to invalid parameter: " << name;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
        return fallback;
    }
    Logger(CHECK) << "Parameter " << name << " checked, set at " << value;
    return value;
}

/**
 * @brief Ensures a value is within `[min, max]`. Returns `fallback` if invalid.
 */
template<typename T>
T checkInRange(T value, T min, T max, T fallback, const std::string& name,
               const std::string& source = "value", bool terminate = false)
{
    Logger(CHECK) << "Checking in range " << name;
    if (value < min || value > max) {
        Logger(WARN) << "Invalid " << name << " from " << source << ": " << value
                     << " not in range [" << min << ", " << max << "]. Resetting to: " << fallback;
        if (terminate) {
            Logger(ERROR) << "Terminating due to invalid parameter: " << name;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
        return fallback;
    }
    Logger(CHECK) << "Parameter " << name << " checked, set at " << value;
    return value;
}

// ------------------------------
// BOOLEAN VALUE CHECKS
// ------------------------------

/**
 * @brief Reads a boolean parameter from a ConfigParser and returns fallback if missing.
 */
inline bool checkBoolValue(ConfigParser& conf, const std::string& key, bool fallback,
                           const std::string& source = "config")
{
    try {
        bool value = conf.getVal<bool>(key);
        Logger(CHECK) << "Using parameter '" << key << "' from " << source << ": " << std::boolalpha << value;
        return value;
    } catch (const std::exception&) {
        Logger(WARN) << "Parameter '" << key << "' missing in " << source << ", using fallback: " << std::boolalpha << fallback;
        return fallback;
    }
}

/**
 * @brief Reads a boolean parameter from CLI (cxxopts) and returns fallback if missing.
 */
inline bool checkBoolValue(const cxxopts::ParseResult& result, const std::string& key, bool fallback)
{
    if (result.count(key)) {
        bool value = result[key].as<bool>();
        Logger(CHECK) << "Using CLI parameter '" << key << "': " << std::boolalpha << value;
        return value;
    } else {
        Logger(WARN) << "CLI parameter '" << key << "' not set, using fallback: " << std::boolalpha << fallback;
        return fallback;
    }
}

// ------------------------------
// FILE AND DIRECTORY CHECKS
// ------------------------------

/**
 * @brief Checks if a file exists and is readable.
 * @return true if available, false otherwise
 * @param terminate Exit program if true and file is missing
 * @param message Optional log message
 */
inline bool checkFileAvailable(const std::string& file, bool terminate = false, const std::string& message = "")
{
    std::ifstream fileStream(file);
    if (fileStream.good()) return true;

    if (terminate) {
        if (!message.empty()) Logger(WARN) << message;
        Logger(ERROR) << "Provided file not available: " << file;
        MPI_Finalize();
        std::exit(EXIT_FAILURE);
    }

    return false;
}

/**
 * @brief Checks if a directory exists and is valid.
 * @return true if directory exists and is valid, false otherwise
 * @param terminate Exit program if true and directory is missing/invalid
 * @param message Optional log message
 */
inline bool checkDirectoryAvailable(const std::string& dir, bool terminate = false, const std::string& message = "")
{
    if (!dir.empty() && dir != DefaultValue<std::string>::value()) {
        if (boost::filesystem::exists(dir)) {
            if (!boost::filesystem::is_directory(dir)) {
                Logger(ERROR) << "Path exists but is not a directory: " << dir;
                if (terminate) { Logger(WARN) << message; MPI_Finalize(); exit(EXIT_FAILURE); }
                return false;
            }
            return true;
        } else {
            Logger(ERROR) << "Directory does not exist: " << dir;
            if (terminate) { Logger(WARN) << message; MPI_Finalize(); exit(EXIT_FAILURE); }
            return false;
        }
    }

    Logger(ERROR) << "No valid directory provided!";
    if (terminate) { Logger(WARN) << message; MPI_Finalize(); exit(EXIT_FAILURE); }
    return false;
}

#endif // MILUPHPC_COMPILE_CHECKS_H
