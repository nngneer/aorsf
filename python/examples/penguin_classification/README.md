# Penguin Classification Example

This example demonstrates using pyaorsf to classify penguin species using the Palmer Penguins dataset. It is the Python equivalent of the R example from the aorsf README.

## Contents

- `penguins.csv` - Palmer Penguins dataset (333 observations, 8 variables)
- `penguin_example.py` - Python script demonstrating classification
- `README.md` - This file

## Setup

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install pyaorsf

**Option A: Install from the aorsf repository (if available)**

```bash
# Navigate to the python directory of the aorsf repo
cd /path/to/aorsf/python
pip install -e .
```

**Option B: Install dependencies manually**

If pyaorsf is not yet published to PyPI, you'll need to build it from the aorsf repository. The package requires:

- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7
- scikit-learn >= 1.0
- Armadillo C++ library (system dependency)

### 3. Install additional dependencies

```bash
pip install pandas scikit-learn
```

## Running the Example

```bash
python penguin_example.py
```

## Expected Output

The script will output:

1. **Dataset information** - Shape, species, features
2. **Model summary** - Similar to the R `print(penguin_fit)` output
3. **Predictions** - First 5 samples with actual vs predicted species
4. **Variable importance** - Feature rankings using ANOVA method
5. **Cross-validation** - 5-fold CV accuracy
6. **Full model** - Results with 100 trees and negation importance

Example output:

```
============================================================
Penguin Classification with Oblique Random Forest
============================================================

Loading Palmer Penguins dataset...
Dataset shape: (333, 8)
Species: ['Adelie', 'Chinstrap', 'Gentoo']

Features: ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
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
           OOB stat type: Accuracy
     Variable importance: anova

-----------------------------------------
```

## R Equivalent

This Python example is equivalent to the R code:

```r
library(aorsf)

penguin_fit <- orsf(data = penguins_orsf,
                    n_tree = 5,
                    formula = species ~ .)

penguin_fit
```

## Dataset

The Palmer Penguins dataset contains measurements for 333 penguins from 3 species:

| Variable | Description |
|----------|-------------|
| species | Penguin species (Adelie, Chinstrap, Gentoo) |
| island | Island in Palmer Archipelago (Biscoe, Dream, Torgersen) |
| bill_length_mm | Bill length in millimeters |
| bill_depth_mm | Bill depth in millimeters |
| flipper_length_mm | Flipper length in millimeters |
| body_mass_g | Body mass in grams |
| sex | Penguin sex (female, male) |
| year | Study year (2007, 2008, 2009) |

Data source: Gorman KB, Williams TD, Fraser WR (2014). Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081.
