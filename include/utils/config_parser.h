/**
 * @file config_parser.h
 * @brief Config parser.
 *
 * Config parser built on top of boost/filesystem and boost/property_tree.
 * Supported file formats are
 *
 * * `.info`
 * * `.json`
 *
 * @author Johannes Martin
 * @author Michael Staneker
 * @bug no known bugs
 * @todo support more file formats
 */
#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

// -----------------------------
// Standard Library
// -----------------------------
#include <iostream>    // For standard output (e.g., std::cout, std::cerr)
#include <string>      // For using std::string
#include <list>        // For using std::list (doubly-linked list from STL)

// -----------------------------
// Boost Libraries
// -----------------------------
#include <boost/filesystem.hpp>                     // For file and directory operations (checking existence, creating paths, etc.)
#include <boost/property_tree/ptree.hpp>            // For the property tree data structure (hierarchical key-value pairs)
#include <boost/property_tree/json_parser.hpp>      // For reading and writing JSON files into/from property trees
#include <boost/property_tree/info_parser.hpp>      // For parsing Boost .info files into property trees
#include <boost/exception/exception.hpp>            // For enhanced exception handling in Boost
#include <boost/current_function.hpp>               // Provides current function name as string (useful for logging/debugging)
#include <boost/throw_exception.hpp>                // For throwing Boost-compatible exceptions (especially when RTTI is disabled)
#include <boost/foreach.hpp>                        // For BOOST_FOREACH macro (pre-C++11 range-based for loops)


//#define BOOST_THROW_EXCEPTION(x) ::boost::throw_exception(x)

/**
 * @brief Config parser class for reading input parameter/settings.
 *
 * A config parser class able to read different input file formats.
 * Currently supported formats are:
 *
 * * `.info`
 * * `.json`
 *
 */
class ConfigParser {
public:

    /**
     * Default constructor.
     */
    ConfigParser();

    /**
     * Constructor.
     *
     * @param file Input config file.
     */
    ConfigParser(const std::string &file);

    /**
     *
     * @param key
     * @return
     */
    std::list<ConfigParser> getObjList(const std::string &key);

    /**
     *
     * @param key
     * @return
     */
    ConfigParser getObj(const std::string &key);

    /**
     *
     * @tparam T
     * @param key
     * @return
     */
    template <typename T>
    T getVal(const std::string &key);

    /**
     *
     * @tparam T
     * @param key
     * @return
     */
    template <typename T>
    std::list<T> getList(const std::string &key);

private:
    // Enable instantiation by ptree to make a recursive usage possible
    /**
     *
     * @param subtree
     */
    ConfigParser(const boost::property_tree::ptree &subtree);

    /// boost property tree instance
    boost::property_tree::ptree tree;
};

/**
 *
 * @tparam T
 * @param key
 * @return
 */
template <typename T>
T ConfigParser::getVal(const std::string &key) {
    return tree.get<T>(key);
}

/**
 *
 * @tparam T
 * @param key
 * @return
 */
template <typename T>
std::list<T> ConfigParser::getList(const std::string &key) {
    std::list <T> lst_;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &val,tree.get_child(key))
    {
        if (val.second.empty()) { // normal list, just fill it with type T
            lst_.push_back(val.second.get_value<T>());
        } else {
            std::cerr << "List does not contain single values. Please use 'getObjList(const std::string &key)'instead."
                         << " - Returning empty list." << std::endl;
        }
    }
    return lst_;
}

#endif //CONFIGPARSER_H
