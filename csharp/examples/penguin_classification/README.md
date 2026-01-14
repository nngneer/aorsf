# Penguin Classification Example

This example demonstrates fitting an oblique random forest classifier to predict penguin species using the Palmer Penguins dataset.

## Dataset

The Palmer Penguins dataset contains measurements of 333 penguins from three species:
- Adelie
- Chinstrap
- Gentoo

Features include bill dimensions, flipper length, body mass, island, sex, and year.

## Running the Example

```bash
cd PenguinExample
dotnet run
```

## Output

The example will:

1. Load and display the dataset summary
2. Fit a small model (5 trees) with ANOVA importance
3. Print model summary similar to R's aorsf output
4. Show predictions for the first 5 observations
5. Display variable importance rankings
6. Fit a larger model (100 trees) with negation importance

## Expected Output

```
============================================================
Penguin Classification with Oblique Random Forest
============================================================

Loading Palmer Penguins dataset...
Dataset shape: (333, 7)
Species: Adelie, Chinstrap, Gentoo

Features: island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year
N observations: 333
N classes: 3
N predictors: 7

------------------------------------------------------------
Fitting Oblique Random Classification Forest...
------------------------------------------------------------

---------- Oblique random classification forest

     Linear combinations: Accelerated Logistic regression
          N observations: 333
               N classes: 3
                 N trees: 5
      N predictors total: 7
   N predictors per node: 3
 Min observations in leaf: 5
          OOB stat value: 0.98
           OOB stat type: AUC-ROC
     Variable importance: anova

-----------------------------------------

============================================================
Predictions
============================================================

First 5 predictions:
Actual       Predicted    Probabilities
--------------------------------------------------
Adelie       Adelie       [1.00, 0.00, 0.00]
Adelie       Adelie       [1.00, 0.00, 0.00]
...
```

## Comparison with Python

This example mirrors the Python version at `python/examples/penguin_classification/penguin_example.py`.

## Reference

Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package version 0.1.0.
