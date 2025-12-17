
<!-- README.md is generated from README.Rmd. Please edit that file -->

# aorsf <a href="https://docs.ropensci.org/aorsf/"><img src="man/figures/logo.png" align="right" height="138" /></a>

<!-- badges: start -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Codecov test
coverage](https://codecov.io/gh/bcjaeger/aorsf/branch/master/graph/badge.svg)](https://app.codecov.io/gh/bcjaeger/aorsf?branch=master)
[![R-CMD-check](https://github.com/bcjaeger/aorsf/workflows/R-CMD-check/badge.svg)](https://github.com/ropensci/aorsf/actions/)
[![Status at rOpenSci Software Peer
Review](https://badges.ropensci.org/532_status.svg)](https://github.com/ropensci/software-review/issues/532/)
<a href="https://joss.theoj.org/papers/10.21105/joss.04705"><img src="https://joss.theoj.org/papers/10.21105/joss.04705/status.svg"></a>
[![CRAN
status](https://www.r-pkg.org/badges/version/aorsf)](https://CRAN.R-project.org/package=aorsf)
[![DOI](https://zenodo.org/badge/394311897.svg)](https://zenodo.org/doi/10.5281/zenodo.7116854)
<!-- badges: end -->

Fit, interpret, and make predictions with oblique random forests (RFs).

## Why aorsf?

- Fast and versatile tools for oblique RFs.<sup>1</sup>

- Accurate predictions.<sup>2</sup>

- Intuitive design with formula based interface.

- Extensive input checks and informative error messages.

- Compatible with `tidymodels` and `mlr3`

## Installation

You can install `aorsf` from CRAN using

``` r
install.packages("aorsf")
```

You can install the development version of aorsf from
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("ropensci/aorsf")
```

## Standalone C++ Library

The core computational engine of `aorsf` is written in standalone C++
with no R dependencies. This allows the library to be used with other
language bindings (e.g., Python).

### Library Structure

    src/
    ├── core/           # Standalone C++ library (no R/Rcpp dependencies)
    │   ├── Forest.cpp/h
    │   ├── Tree.cpp/h
    │   ├── Data.h
    │   ├── Coxph.cpp/h
    │   ├── utility.cpp/h
    │   └── ...
    └── rcpp/           # R-specific adapters
        ├── RcppOutput.h
        ├── RcppInterrupts.h
        └── RcppRMath.h

### Dependencies

The core library requires:

- **Armadillo** (C++ linear algebra library)
- A C++17 compatible compiler

### Building for Other Languages

The core library uses dependency injection for platform-specific
functionality:

- **Output**: Implement `OutputHandler` interface for logging
- **Interrupts**: Implement `InterruptHandler` interface for user
  interrupt checking
- **Statistics**: Implement `StatDistributions` interface for `pt()` and
  `pchisq()` functions

Example initialization for a custom binding:

``` cpp
#include "core/Forest.h"
#include "core/Output.h"
#include "core/Interrupts.h"
#include "core/RMath.h"

// Set up custom handlers before using the library
aorsf::OutputManager::set_handler(my_output_handler);
aorsf::InterruptManager::set_handler(my_interrupt_handler);
aorsf::StatManager::set_distributions(my_stat_distributions);
```

For Python bindings, consider using
[nanobind](https://github.com/wjakob/nanobind) with
[carma](https://github.com/RUrlus/carma) for Armadillo/NumPy
interoperability.

## Get started

``` r
library(aorsf)
library(tidyverse)
```

`aorsf` fits several types of oblique RFs with the `orsf()` function,
including classification, regression, and survival RFs.

For classification, we fit an oblique RF to predict penguin species
using `penguin` data from the magnificent `palmerpenguins` [R
package](https://allisonhorst.github.io/palmerpenguins/)

``` r
# An oblique classification RF
penguin_fit <- orsf(data = penguins_orsf,
                    n_tree = 5, 
                    formula = species ~ .)

penguin_fit
#> ---------- Oblique random classification forest
#> 
#>      Linear combinations: Accelerated Logistic regression
#>           N observations: 333
#>                N classes: 3
#>                  N trees: 5
#>       N predictors total: 7
#>    N predictors per node: 3
#>  Average leaves per tree: 5.8
#> Min observations in leaf: 5
#>           OOB stat value: 0.98
#>            OOB stat type: AUC-ROC
#>      Variable importance: anova
#> 
#> -----------------------------------------
```

For regression, we use the same data but predict bill length of
penguins:

``` r
# An oblique regression RF
bill_fit <- orsf(data = penguins_orsf, 
                 n_tree = 5, 
                 formula = bill_length_mm ~ .)

bill_fit
#> ---------- Oblique random regression forest
#> 
#>      Linear combinations: Accelerated Linear regression
#>           N observations: 333
#>                  N trees: 5
#>       N predictors total: 7
#>    N predictors per node: 3
#>  Average leaves per tree: 51.2
#> Min observations in leaf: 5
#>           OOB stat value: 0.77
#>            OOB stat type: RSQ
#>      Variable importance: anova
#> 
#> -----------------------------------------
```

My personal favorite is the oblique survival RF with accelerated Cox
regression because it was the first type of oblique RF that `aorsf`
provided (see [ArXiv paper](https://arxiv.org/abs/2208.01129); the paper
is also published in *Journal of Computational and Graphical Statistics*
but is not publicly available there). Here, we use it to predict
mortality risk following diagnosis of primary biliary cirrhosis:

``` r
# An oblique survival RF
pbc_fit <- orsf(data = pbc_orsf, 
                n_tree = 5,
                formula = Surv(time, status) ~ . - id)

pbc_fit
#> ---------- Oblique random survival forest
#> 
#>      Linear combinations: Accelerated Cox regression
#>           N observations: 276
#>                 N events: 111
#>                  N trees: 5
#>       N predictors total: 17
#>    N predictors per node: 5
#>  Average leaves per tree: 20.8
#> Min observations in leaf: 5
#>       Min events in leaf: 1
#>           OOB stat value: 0.78
#>            OOB stat type: Harrell's C-index
#>      Variable importance: anova
#> 
#> -----------------------------------------
```

## What does “oblique” mean?

Decision trees are grown by splitting a set of training data into
non-overlapping subsets, with the goal of having more similarity within
the new subsets than between them. When subsets are created with a
single predictor, the decision tree is *axis-based* because the subset
boundaries are perpendicular to the axis of the predictor. When linear
combinations (i.e., a weighted sum) of variables are used instead of a
single variable, the tree is *oblique* because the boundaries are
neither parallel nor perpendicular to the axis.

**Figure**: Decision trees for classification with axis-based splitting
(left) and oblique splitting (right). Cases are orange squares; controls
are purple circles. Both trees partition the predictor space defined by
variables X1 and X2, but the oblique splits do a better job of
separating the two classes.

<img src="man/figures/tree_axis_v_oblique.png" width="100%" />

So, how does this difference translate to real data, and how does it
impact random forests comprising hundreds of axis-based or oblique
trees? We will demonstrate this using the `penguin` data.<sup>3</sup> We
will also use this function to make several plots:

``` r
plot_decision_surface <- function(predictions, title, grid){
 
 # this is not a general function for plotting
 # decision surfaces. It just helps to minimize 
 # copying and pasting of code.
 
 class_preds <- bind_cols(grid, predictions) %>%
  pivot_longer(cols = c(Adelie,
                        Chinstrap,
                        Gentoo)) %>%
  group_by(flipper_length_mm, bill_length_mm) %>%
  arrange(desc(value)) %>%
  slice(1)
 
 cols <- c("darkorange", "purple", "cyan4")

 ggplot(class_preds, aes(bill_length_mm, flipper_length_mm)) +
  geom_contour_filled(aes(z = value, fill = name),
                      alpha = .25) +
  geom_point(data = penguins_orsf,
             aes(color = species, shape = species),
             alpha = 0.5) +
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  labs(x = "Bill length, mm",
       y = "Flipper length, mm") +
  theme_minimal() +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  theme(panel.grid = element_blank(),
        panel.border = element_rect(fill = NA),
        legend.position = '') + 
  labs(title = title)
 
}
```

We also use a grid of points for plotting decision surfaces:

``` r
grid <- expand_grid(

 flipper_length_mm = seq(min(penguins_orsf$flipper_length_mm),
                     max(penguins_orsf$flipper_length_mm),
                  len = 200),
 bill_length_mm = seq(min(penguins_orsf$bill_length_mm),
                      max(penguins_orsf$bill_length_mm),
                      len = 200)
)
```

We use `orsf` with `mtry=1` to fit axis-based trees:

``` r
fit_axis_tree <- penguins_orsf %>% 
 orsf(species ~ bill_length_mm + flipper_length_mm,
      n_tree = 1,
      mtry = 1,
      tree_seeds = 106760)
```

Next we use `orsf_update` to copy and modify the original model,
expanding it to fit an oblique tree by using `mtry=2` instead of
`mtry=1`, and to include 500 trees instead of 1:

``` r
fit_axis_forest <- fit_axis_tree %>% 
 orsf_update(n_tree = 500)

fit_oblique_tree <- fit_axis_tree %>% 
 orsf_update(mtry = 2)

fit_oblique_forest <- fit_oblique_tree %>% 
 orsf_update(n_tree = 500)
```

And now we have all we need to visualize decision surfaces using
predictions from these four fits:

``` r
preds <- list(fit_axis_tree,
              fit_axis_forest,
              fit_oblique_tree,
              fit_oblique_forest) %>% 
 map(predict, new_data = grid, pred_type = 'prob')

titles <- c("Axis-based tree",
            "Axis-based forest",
            "Oblique tree",
            "Oblique forest")

plots <- map2(preds, titles, plot_decision_surface, grid = grid)
```

**Figure**: Axis-based and oblique decision surfaces from a single tree
and an ensemble of 500 trees. Axis-based trees have boundaries
perpendicular to predictor axes, whereas oblique trees can have
boundaries that are neither parallel nor perpendicular to predictor
axes. Axis-based forests tend to have ‘step-function’ decision
boundaries, while oblique forests tend to have smooth decision
boundaries.

<img src="man/figures/README-unnamed-chunk-9-1.png" width="100%" />

## Variable importance

The importance of individual predictor variables can be estimated in
three ways using `aorsf` and can be used on any type of oblique RF.
Also, variable importance functions always return a named character
vector

- **negation**<sup>2</sup>: Each variable is assessed separately by
  multiplying the variable’s coefficients by -1 and then determining how
  much the model’s performance changes. The worse the model’s
  performance after negating coefficients for a given variable, the more
  important the variable. This technique is promising b/c it does not
  require permutation and it emphasizes variables with larger
  coefficients in linear combinations, but it is also relatively new and
  hasn’t been studied as much as permutation importance. See
  Jaeger, (2023) for more details on this technique.

  ``` r
  orsf_vi_negate(pbc_fit)
  #>          bili          chol        copper       ascites           age 
  #>  0.0927268221  0.0608229584  0.0478810740  0.0291988056  0.0246337315 
  #>        hepato         stage       spiders       albumin       protime 
  #>  0.0218506525  0.0193156243  0.0125759501  0.0110979392  0.0080810921 
  #>           ast      platelet          trig           sex         edema 
  #>  0.0076885672  0.0068084464  0.0021303266  0.0010149493 -0.0004657918 
  #>      alk.phos           trt 
  #> -0.0058280945 -0.0135098286
  ```

- **permutation**: Each variable is assessed separately by randomly
  permuting the variable’s values and then determining how much the
  model’s performance changes. The worse the model’s performance after
  permuting the values of a given variable, the more important the
  variable. This technique is flexible, intuitive, and frequently used.
  It also has several [known
  limitations](https://christophm.github.io/interpretable-ml-book/feature-importance.html#disadvantages-9)

  ``` r
  orsf_vi_permute(penguin_fit)
  #> flipper_length_mm    bill_length_mm       body_mass_g            island 
  #>       0.147092756       0.101879429       0.096925209       0.077248082 
  #>     bill_depth_mm               sex              year 
  #>       0.045489860       0.010025729      -0.001415467
  ```

- **analysis of variance (ANOVA)**<sup>4</sup>: A p-value is computed
  for each coefficient in each linear combination of variables in each
  decision tree. Importance for an individual predictor variable is the
  proportion of times a p-value for its coefficient is \< 0.01. This
  technique is very efficient computationally, but may not be as
  effective as permutation or negation in terms of selecting signal over
  noise variables. See [Menze,
  2011](https://link.springer.com/chapter/10.1007/978-3-642-23783-6_29)
  for more details on this technique.

  ``` r
  orsf_vi_anova(bill_fit)
  #>           species               sex flipper_length_mm            island 
  #>        0.24618736        0.14545455        0.08620690        0.06499109 
  #>     bill_depth_mm       body_mass_g              year 
  #>        0.06153846        0.05468750        0.00000000
  ```

You can supply your own R function to estimate out-of-bag error (see
[oob vignette](https://docs.ropensci.org/aorsf/articles/oobag.html)) or
to estimate out-of-bag variable importance (see [orsf_vi
examples](https://docs.ropensci.org/aorsf/reference/orsf_vi.html#examples))

## Partial dependence (PD)

Partial dependence (PD) shows the expected prediction from a model as a
function of a single predictor or multiple predictors. The expectation
is marginalized over the values of all other predictors, giving
something like a multivariable adjusted estimate of the model’s
prediction.. You can use specific values for a predictor to compute PD
or let `aorsf` pick reasonable values for you if you use
`pred_spec_auto()`:

``` r
# pick your own values
orsf_pd_oob(bill_fit, pred_spec = list(species = c("Adelie", "Gentoo")))
#>    species     mean      lwr     medn      upr
#>     <fctr>    <num>    <num>    <num>    <num>
#> 1:  Adelie 41.74289 36.00233 40.70375 51.98937
#> 2:  Gentoo 43.38986 36.42250 43.20000 50.91862

# let aorsf pick reasonable values for you:
orsf_pd_oob(bill_fit, pred_spec = pred_spec_auto(bill_depth_mm, island))
#>     bill_depth_mm    island     mean      lwr     medn      upr
#>             <num>    <fctr>    <num>    <num>    <num>    <num>
#>  1:         14.32    Biscoe 45.31979 36.06667 46.00714 50.00000
#>  2:         15.60    Biscoe 44.68833 36.03333 45.03333 51.85606
#>  3:         17.30    Biscoe 43.95395 35.97108 44.17619 53.72679
#>  4:         18.70    Biscoe 44.41456 36.09233 44.86000 54.69222
#>  5:         19.50    Biscoe 44.50362 36.14572 44.86000 54.69222
#>  6:         14.32     Dream 45.20036 37.45500 45.76389 50.04442
#>  7:         15.60     Dream 44.94930 36.53667 45.49476 52.96000
#>  8:         17.30     Dream 44.06989 36.21500 44.17619 53.40000
#>  9:         18.70     Dream 44.58579 36.23099 44.86000 53.88649
#> 10:         19.50     Dream 44.67921 36.20874 45.33000 53.85375
#> 11:         14.32 Torgersen 44.38289 36.06952 45.33000 50.92758
#> 12:         15.60 Torgersen 44.43864 36.10000 44.97500 50.94286
#> 13:         17.30 Torgersen 44.03883 35.94000 43.71250 53.72679
#> 14:         18.70 Torgersen 44.29852 35.80000 44.57286 54.69222
#> 15:         19.50 Torgersen 44.29565 35.80000 44.47111 54.69222
```

The summary function, `orsf_summarize_uni()`, computes PD for as many
variables as you ask it to, using sensible values.

``` r
orsf_summarize_uni(pbc_fit, n_variables = 2)
#> 
#> -- bili (VI Rank: 1) ----------------------------
#> 
#>         |---------------- Risk ----------------|
#>   Value      Mean    Median     25th %    75th %
#>  <char>     <num>     <num>      <num>     <num>
#>    0.60 0.2652584 0.1666667 0.03213938 0.4132576
#>    0.80 0.2619876 0.1557492 0.03078482 0.4073864
#>    1.40 0.2856810 0.1855346 0.03333333 0.4833333
#>    3.52 0.4076895 0.3333333 0.04471801 0.6919147
#>    7.25 0.5296084 0.5166667 0.14829505 0.8822115
#> 
#> -- chol (VI Rank: 2) ----------------------------
#> 
#>         |---------------- Risk ----------------|
#>   Value      Mean    Median     25th %    75th %
#>  <char>     <num>     <num>      <num>     <num>
#>     212 0.3572797 0.1982323 0.02504209 0.6666667
#>     250 0.3546476 0.1982323 0.02504209 0.6666667
#>     310 0.3515586 0.2147727 0.03846154 0.5826599
#>     401 0.3619913 0.2475329 0.05229197 0.5826599
#>     567 0.3834873 0.3021991 0.09090909 0.6638889
#> 
#>  Predicted risk at time t = 1788 for top 2 predictors
```

For more on PD, see the
[vignette](https://docs.ropensci.org/aorsf/articles/pd.html)

## Individual conditional expectations (ICE)

Unlike partial dependence, which shows the expected prediction as a
function of one or multiple predictors, individual conditional
expectations (ICE) show the prediction for an individual observation as
a function of a predictor.

For more on ICE, see the
[vignette](https://docs.ropensci.org/aorsf/articles/pd.html#individual-conditional-expectations-ice)

## Interaction scores

The `orsf_vint()` function computes a score for each possible
interaction in a model based on PD using the method described in
Greenwell et al, 2018.<sup>5</sup> It can be slow for larger datasets,
but substantial speedups occur by making use of multi-threading and
restricting the search to a smaller set of predictors.

``` r
preds_interaction <- c("albumin", "protime", "bili", "spiders", "trt")

# While it is tempting to speed up `orsf_vint()` by growing a smaller 
# number of trees, results may become unstable with this shortcut.
pbc_interactions <- pbc_fit %>% 
 orsf_update(n_tree = 500, tree_seeds = 329) %>% 
 orsf_vint(n_thread = 0,  predictors = preds_interaction)

pbc_interactions
#>          interaction     score          pd_values
#>               <char>     <num>             <list>
#>  1: albumin..protime 1.0725123 <data.table[25x9]>
#>  2:    bili..spiders 0.8458761 <data.table[10x9]>
#>  3: protime..spiders 0.8456060 <data.table[10x9]>
#>  4:    protime..bili 0.7119054 <data.table[25x9]>
#>  5:    albumin..bili 0.5468455 <data.table[25x9]>
#>  6: albumin..spiders 0.3183702 <data.table[10x9]>
#>  7:        bili..trt 0.2500514 <data.table[10x9]>
#>  8:     spiders..trt 0.1456292  <data.table[4x9]>
#>  9:     albumin..trt 0.1180540 <data.table[10x9]>
#> 10:     protime..trt 0.0907605 <data.table[10x9]>
```

What do the values in `score` mean? These values are the average of the
standard deviation of the standard deviation of PD in one variable
conditional on the other variable. They should be interpreted relative
to one another, i.e., a higher scoring interaction is more likely to
reflect a real interaction between two variables than a lower scoring
one.

Do these interaction scores make sense? Let’s test the top scoring and
lowest scoring interactions using `coxph()`.

``` r
library(survival)
# the top scoring interaction should get a lower p-value
anova(coxph(Surv(time, status) ~ protime * albumin, data = pbc_orsf))
#> Analysis of Deviance Table
#>  Cox model: response is Surv(time, status)
#> Terms added sequentially (first to last)
#> 
#>                  loglik  Chisq Df Pr(>|Chi|)    
#> NULL            -550.19                         
#> protime         -538.51 23.353  1  1.349e-06 ***
#> albumin         -514.89 47.255  1  6.234e-12 ***
#> protime:albumin -511.76  6.252  1    0.01241 *  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# the bottom scoring interaction should get a higher p-value
anova(coxph(Surv(time, status) ~ spiders * trt, data = pbc_orsf))
#> Analysis of Deviance Table
#>  Cox model: response is Surv(time, status)
#> Terms added sequentially (first to last)
#> 
#>              loglik   Chisq Df Pr(>|Chi|)    
#> NULL        -550.19                          
#> spiders     -538.58 23.2159  1  1.448e-06 ***
#> trt         -538.39  0.3877  1     0.5335    
#> spiders:trt -538.29  0.2066  1     0.6494    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Note: this is exploratory and not a true null hypothesis test. Why?
Because we used the same data both to generate and to test the null
hypothesis. We are not so much conducting statistical inference when we
test these interactions with `coxph` as we are demonstrating the
interaction scores that `orsf_vint()` provides are consistent with tests
from other models.

## Comparison to existing software

For survival analysis, comparisons between `aorsf` and existing software
are presented in our [JCGS
paper](https://doi.org/10.1080/10618600.2023.2231048). The paper:

- describes `aorsf` in detail with a summary of the procedures used in
  the tree fitting algorithm

- runs a general benchmark comparing `aorsf` with `obliqueRSF` and
  several other learners

- reports prediction accuracy and computational efficiency of all
  learners.

- runs a simulation study comparing variable importance techniques with
  oblique survival RFs, axis based survival RFs, and boosted trees.

- reports the probability that each variable importance technique will
  rank a relevant variable with higher importance than an irrelevant
  variable.

<!-- A more hands-on comparison of `aorsf` and other R packages is provided in [orsf examples](https://docs.ropensci.org/aorsf/reference/orsf.html#tidymodels) -->

## References

1.  Jaeger BC, Long DL, Long DM, Sims M, Szychowski JM, Min Y, Mcclure
    LA, Howard G, Simon N (2019). “Oblique random survival forests.”
    *The Annals of Applied Statistics*, *13*(3).
2.  Jaeger BC, Welden S, Lenoir K, Speiser JL, Segar MW, Pandey A,
    Pajewski NM (2023). “Accelerated and interpretable oblique random
    survival forests.” *Journal of Computational and Graphical
    Statistics*, 1-16.
3.  Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer
    Archipelago (Antarctica) penguin data*. R package version 0.1.0,
    <https://allisonhorst.github.io/palmerpenguins/>.
4.  Menze, H B, Kelm, Michael B, Splitthoff, N D, Koethe, Ullrich,
    Hamprecht, A F (2011). “On oblique random forests.” In *Machine
    Learning and Knowledge Discovery in Databases: European Conference,
    ECML PKDD 2011, Athens, Greece, September 5-9, 2011, Proceedings,
    Part II 22*, 453-469. Springer.
5.  Greenwell, M B, Boehmke, C B, McCarthy, J A (2018). “A simple and
    effective model-based variable importance measure.” *arXiv preprint
    arXiv:1805.04755*.

## Funding

The developers of `aorsf` received financial support from the Center for
Biomedical Informatics, Wake Forest University School of Medicine. We
also received support from the National Center for Advancing
Translational Sciences of the National Institutes of Health under Award
Number UL1TR001420.

The content is solely the responsibility of the authors and does not
necessarily represent the official views of the National Institutes of
Health.
