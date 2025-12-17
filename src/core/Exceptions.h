/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_EXCEPTIONS_H_
#define AORSF_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace aorsf {

/**
 * @brief Base exception class for aorsf errors.
 *
 * All aorsf-specific exceptions inherit from this class,
 * allowing callers to catch all aorsf errors with a single handler.
 */
class aorsf_error : public std::runtime_error {
public:
    explicit aorsf_error(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @brief Exception for invalid argument errors.
 *
 * Thrown when function arguments don't meet requirements.
 */
class invalid_argument_error : public aorsf_error {
public:
    explicit invalid_argument_error(const std::string& msg)
        : aorsf_error(msg) {}
};

/**
 * @brief Exception for computation errors.
 *
 * Thrown when a computation fails (e.g., empty data, convergence failure).
 */
class computation_error : public aorsf_error {
public:
    explicit computation_error(const std::string& msg)
        : aorsf_error(msg) {}
};

} // namespace aorsf

#endif // AORSF_EXCEPTIONS_H_
