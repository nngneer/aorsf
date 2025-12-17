/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_OUTPUT_H_
#define AORSF_OUTPUT_H_

#include <string>
#include <memory>
#include <sstream>

namespace aorsf {

/**
 * @brief Abstract interface for console output.
 *
 * This allows the core library to output progress messages without
 * depending on R/Rcpp. Different implementations can be provided
 * for R (using Rcpp::Rcout) or standalone builds (using std::cout).
 */
class OutputHandler {
public:
    virtual ~OutputHandler() = default;
    virtual void print(const std::string& msg) = 0;
    virtual void println(const std::string& msg) = 0;
};

/**
 * @brief Silent output handler that discards all output.
 *
 * This is the default implementation used when no handler is set,
 * ensuring the library works silently by default.
 */
class SilentOutput : public OutputHandler {
public:
    void print(const std::string&) override {}
    void println(const std::string&) override {}
};

/**
 * @brief Global output manager for accessing the current output handler.
 *
 * Usage:
 *   OutputManager::get().println("Growing trees: 50%");
 *
 * Or with the convenience macro:
 *   AORSF_OUT.println("Growing trees: 50%");
 */
class OutputManager {
public:
    static void set_handler(std::shared_ptr<OutputHandler> handler) {
        get_handler_ptr() = handler ? handler : std::make_shared<SilentOutput>();
    }

    static OutputHandler& get() {
        return *get_handler_ptr();
    }

private:
    // Use a function-local static to avoid static initialization order issues
    // and avoid the need for a separate .cpp file
    static std::shared_ptr<OutputHandler>& get_handler_ptr() {
        static std::shared_ptr<OutputHandler> handler = std::make_shared<SilentOutput>();
        return handler;
    }
};

// Convenience macro for output
#define AORSF_OUT aorsf::OutputManager::get()

} // namespace aorsf

#endif // AORSF_OUTPUT_H_
