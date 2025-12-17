/*-----------------------------------------------------------------------------
 This file is part of aorsf.
 Author: Byron C Jaeger
 aorsf may be modified and distributed under the terms of the MIT license.
#----------------------------------------------------------------------------*/

#ifndef AORSF_ARMA_CONFIG_H_
#define AORSF_ARMA_CONFIG_H_

// Configure Armadillo to use 32-bit words for compatibility with R/RcppArmadillo
// This must be defined before including armadillo
#if !defined(ARMA_64BIT_WORD)
  #define ARMA_32BIT_WORD 1
#endif

#include <armadillo>

#endif // AORSF_ARMA_CONFIG_H_
