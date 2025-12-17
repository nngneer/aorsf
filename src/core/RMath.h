/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_RMATH_H_
#define AORSF_RMATH_H_

#include <cmath>
#include <limits>
#include <functional>
#include <memory>

namespace aorsf {

// Constants replacing R_PosInf and R_NegInf
constexpr double AORSF_POS_INF = std::numeric_limits<double>::infinity();
constexpr double AORSF_NEG_INF = -std::numeric_limits<double>::infinity();

/**
 * @brief Abstract interface for statistical distribution functions.
 *
 * This allows the core library to compute p-values without
 * depending on R/Rcpp. Different implementations can be provided
 * for R (using R::pt, R::pchisq) or standalone builds.
 */
class StatDistributions {
public:
    virtual ~StatDistributions() = default;

    /**
     * @brief Compute the CDF of the t-distribution.
     * @param x quantile
     * @param df degrees of freedom
     * @return P(T <= x) where T follows t(df)
     */
    virtual double pt(double x, double df) = 0;

    /**
     * @brief Compute the CDF of the chi-squared distribution.
     * @param x quantile
     * @param df degrees of freedom
     * @return P(X <= x) where X follows chi-squared(df)
     */
    virtual double pchisq(double x, double df) = 0;
};

/**
 * @brief Default implementation using standard normal approximation.
 * This provides basic functionality when no R backend is available.
 */
class DefaultStatDistributions : public StatDistributions {
public:
    double pt(double x, double df) override {
        // For large df, t-distribution approximates normal
        // This is a rough approximation
        if (df > 30) {
            return pnorm(x);
        }
        // For smaller df, use a simple approximation
        // This is less accurate but functional
        double t_adj = x * std::sqrt(1.0 - 1.0 / (4.0 * df));
        return pnorm(t_adj);
    }

    double pchisq(double x, double df) override {
        // Wilson-Hilferty approximation for chi-squared CDF
        if (x <= 0) return 0.0;
        double z = std::pow(x / df, 1.0/3.0) - (1.0 - 2.0 / (9.0 * df));
        z /= std::sqrt(2.0 / (9.0 * df));
        return pnorm(z);
    }

private:
    // Standard normal CDF approximation
    double pnorm(double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }
};

/**
 * @brief Global manager for statistical distribution functions.
 */
class StatManager {
public:
    static void set_distributions(std::shared_ptr<StatDistributions> dist) {
        get_dist_ptr() = dist ? dist : std::make_shared<DefaultStatDistributions>();
    }

    static StatDistributions& get() {
        return *get_dist_ptr();
    }

private:
    static std::shared_ptr<StatDistributions>& get_dist_ptr() {
        static std::shared_ptr<StatDistributions> dist = std::make_shared<DefaultStatDistributions>();
        return dist;
    }
};

// Convenience macro
#define AORSF_STAT aorsf::StatManager::get()

} // namespace aorsf

#endif // AORSF_RMATH_H_
