//
// Created by Christian Jetter on 09.09.25.
//

#ifndef MILUPHPC_TYPES_H
#define MILUPHPC_TYPES_H

/**
 * Type definitions
 * * `real` corresponds to floating point precision for whole program
 * * `keyType` influences the maximal tree depth
 *     * maximal tree depth: (sizeof(keyType) - (sizeof(keyType) % DIM))/DIM
 */
#ifdef SINGLE_PRECISION
typedef float real;
#else
typedef double real;
#endif
typedef int integer;
typedef unsigned long keyType;
typedef int idInteger;

#endif //MILUPHPC_TYPES_H
