/*-----------------------------------------------------------------------------
 Python interrupt handler for pyaorsf.
 Checks for Python signals (Ctrl+C) during long-running operations.
#----------------------------------------------------------------------------*/

#ifndef PYAORSF_PYTHON_INTERRUPTS_H_
#define PYAORSF_PYTHON_INTERRUPTS_H_

#include <nanobind/nanobind.h>
#include "Interrupts.h"

namespace nb = nanobind;

namespace aorsf {

/**
 * @brief Python interrupt handler using PyErr_CheckSignals
 */
class PythonInterrupt : public InterruptHandler {
public:
    bool check() override {
        nb::gil_scoped_acquire guard;
        if (PyErr_CheckSignals() != 0) {
            PyErr_Clear();  // Clear the error, we'll handle it
            return true;
        }
        return false;
    }
};

/**
 * @brief Initialize Python interrupt handler.
 */
inline void init_python_interrupt() {
    InterruptManager::set_handler(std::make_shared<PythonInterrupt>());
}

} // namespace aorsf

#endif // PYAORSF_PYTHON_INTERRUPTS_H_
