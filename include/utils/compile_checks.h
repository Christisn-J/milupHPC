//
// Created by Christian Jetter on 09.09.25.
// Purpose: Runtime checks for configuration values with fallback logic
//

#ifndef MILUPHPC_COMPILE_CHECKS_H
#define MILUPHPC_COMPILE_CHECKS_H

#include "logger.h"
#include "../constants.h"

#include <string>     // std::string
#include <fstream>    // std::ifstream
#include <cstdlib>    // exit()
#include <mpi.h>      // MPI_Finalize()

// ------------------------------
// Value Constraint Checks
// ------------------------------

/**
 * Ensures a value is at least 'min'. If not, logs a warning and returns fallback.
 */
template<typename T>
T checkMinValue(T value, T min, T fallback, const std::string& name,
                const std::string& source = "value", bool terminate = false)
{
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
    return value;
}

/**
 * Ensures a value is at most 'max'. If not, logs a warning and returns fallback.
 */
template<typename T>
T checkMaxValue(T value, T max, T fallback, const std::string& name,
                const std::string& source = "value", bool terminate = false)
{
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
    return value;
}

/**
 * Ensures a value is within [min, max]. If not, logs a warning and returns fallback.
 */
template<typename T>
T checkInRange(T value, T min, T max, T fallback, const std::string& name,
               const std::string& source = "value", bool terminate = false)
{
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
    return value;
}

bool checkBoolValue(ConfigParser& conf, const std::string& key, bool fallback, const std::string& source = "config")
{
    try {
        bool value = conf.getVal<bool>(key);
        Logger(INFO) << "Using parameter '" << key << "' from " << source << ": " << std::boolalpha << value;
        return value;
    } catch (const std::exception& e) {
        Logger(WARN) << "Parameter '" << key << "' missing in " << source << ", using fallback: " << std::boolalpha << fallback;
        return fallback;
    }
}

bool checkBoolValue(const cxxopts::ParseResult& result, const std::string& key, bool fallback)
{
    if (result.count(key)) {
        bool value = result[key].as<bool>();
        Logger(INFO) << "Using CLI parameter '" << key << "': " << std::boolalpha << value;
        return value;
    } else {
        Logger(WARN) << "CLI parameter '" << key << "' not set, using fallback: " << std::boolalpha << fallback;
        return fallback;
    }
}



// ------------------------------
// File Availability Check
// ------------------------------

/**
 * Checks if a file exists and is readable. If not, optionally logs and terminates.
 * @param file Path to file
 * @param terminate If true, the program exits on failure
 * @param message Optional additional log message
 * @return true if file is available, false otherwise
 */
inline bool checkFileAvailable(const std::string& file, bool terminate = false, const std::string& message = "")
{
    std::ifstream fileStream(file);
    if (fileStream.good()) {
        return true;
    }

    if (terminate) {
        if (!message.empty()) {
            Logger(WARN) << message;
        }
        Logger(ERROR) << "Provided file not available: " << file;
        MPI_Finalize();
        std::exit(EXIT_FAILURE);
    }

    return false;
}


/**
 * Checks if a directory exists and is valid. If not, optionally logs and terminates.
 * @param dir Path to directory
 * @param terminate If true, the program exits on failure
 * @param message Optional additional log message
 * @return true if directory exists and is a directory, false otherwise
 */
inline bool checkDirectoryAvailable(const std::string& dir, bool terminate = false, const std::string& message = "")
{
    if (!dir.empty() && dir != DefaultValue<std::string>::value()) {
        if (boost::filesystem::exists(dir)) {
            if (!boost::filesystem::is_directory(dir)) {
                Logger(ERROR) << "Path exists but is not a directory: " << dir;
                if (terminate) {
                    Logger(WARN) << message;
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
                return false;
            }
            return true; // Directory exists und ist gültig
        } else {
            Logger(ERROR) << "Directory does not exist: " << dir;
            if (terminate) {
                Logger(WARN) << message;
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
            return false;
        }
    }

    Logger(ERROR) << "No valid directory provided!";
    if (terminate) {
        Logger(WARN) << message;
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    return false;
}

#endif // MILUPHPC_COMPILE_CHECKS_H
