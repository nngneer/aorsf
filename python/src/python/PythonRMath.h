/*-----------------------------------------------------------------------------
 Python statistical functions for pyaorsf.
 Uses scipy.stats for accurate distribution functions.
#----------------------------------------------------------------------------*/

#ifndef PYAORSF_PYTHON_RMATH_H_
#define PYAORSF_PYTHON_RMATH_H_

#include <nanobind/nanobind.h>
#include "RMath.h"

namespace nb = nanobind;

namespace aorsf {

/**
 * @brief Python stat distributions using scipy.stats
 */
class PythonStatDistributions : public StatDistributions {
public:
    double pt(double x, double df) override {
        nb::gil_scoped_acquire guard;
        try {
            nb::module_ stats = nb::module_::import_("scipy.stats");
            nb::object t_dist = stats.attr("t");
            nb::object result = t_dist.attr("cdf")(x, df);
            return nb::cast<double>(result);
        } catch (...) {
            // Fallback to default approximation
            DefaultStatDistributions fallback;
            return fallback.pt(x, df);
        }
    }

    double pchisq(double x, double df) override {
        nb::gil_scoped_acquire guard;
        try {
            nb::module_ stats = nb::module_::import_("scipy.stats");
            nb::object chi2_dist = stats.attr("chi2");
            nb::object result = chi2_dist.attr("cdf")(x, df);
            return nb::cast<double>(result);
        } catch (...) {
            // Fallback to default approximation
            DefaultStatDistributions fallback;
            return fallback.pchisq(x, df);
        }
    }
};

/**
 * @brief Initialize Python stat distributions.
 */
inline void init_python_stat() {
    StatManager::set_distributions(std::make_shared<PythonStatDistributions>());
}

} // namespace aorsf

#endif // PYAORSF_PYTHON_RMATH_H_
