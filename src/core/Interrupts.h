/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_INTERRUPTS_H_
#define AORSF_INTERRUPTS_H_

#include <functional>
#include <memory>
#include <atomic>

namespace aorsf {

/**
 * @brief Abstract interface for checking user interrupts.
 *
 * This allows the core library to check for interrupts without
 * depending on R/Rcpp. Different implementations can be provided
 * for R (using Rcpp::checkUserInterrupt) or Python (using PyErr_CheckSignals).
 */
class InterruptHandler {
public:
    virtual ~InterruptHandler() = default;

    /**
     * @brief Check if the user has requested an interrupt.
     * @return true if interrupted, false otherwise
     */
    virtual bool check() = 0;
};

/**
 * @brief Default interrupt handler that never interrupts.
 */
class NoInterrupt : public InterruptHandler {
public:
    bool check() override { return false; }
};

/**
 * @brief Global interrupt manager for accessing the current handler.
 */
class InterruptManager {
public:
    static void set_handler(std::shared_ptr<InterruptHandler> handler) {
        get_handler_ptr() = handler ? handler : std::make_shared<NoInterrupt>();
    }

    static bool check() {
        return get_handler_ptr()->check();
    }

private:
    static std::shared_ptr<InterruptHandler>& get_handler_ptr() {
        static std::shared_ptr<InterruptHandler> handler = std::make_shared<NoInterrupt>();
        return handler;
    }
};

// Convenience macro
#define AORSF_CHECK_INTERRUPT() if (aorsf::InterruptManager::check()) return

} // namespace aorsf

#endif // AORSF_INTERRUPTS_H_
