# Statistics Computing Consolidation Project

Hello there,

This repository is organized as a **single statistics computing project**. The goal is to show how simulation, estimation, regression, asymptotic reasoning, and visualization can be integrated into one reproducible Python workflow.

## Project idea

The script consolidates several computational exercises that were originally written in R and rebuilds them into one Python program with a cleaner structure for GitHub. Each section of the script follows the same logic:

1. **Question and purpose**  
   What statistical problem is being studied.

2. **What the code does**  
   The simulation, estimation, or empirical procedure used.

3. **Outcome**  
   The numerical results and visual evidence produced by the script.

This makes the code easier to read for someone who did not see the original assignments.

## Main components

The script covers five broad tasks:

### 1. Distribution construction from a common source of randomness
Starting from one `Uniform(0,1)` sample, the script constructs observations from several distributions and compares their empirical behavior graphically.

### 2. Dependence and conditional analysis in simulated multivariate data
The script simulates bivariate normal data, studies conditional means across quantile groups, checks empirical expectation and variance decompositions, reverses the sign of correlation, and then repeats the analysis for a lognormal transformation.

### 3. Descriptive analysis and linear regression
Using the classical `cars` dataset, the script computes descriptive statistics, estimates a simple linear regression, studies residuals, and runs simulation exercises that show how coefficient stability changes when new variables are generated from the fitted structure.

### 4. Large-sample behavior and estimator comparison
The code illustrates the Law of Large Numbers and the Central Limit Theorem through repeated simulation, then compares Method of Moments and Maximum Likelihood in a restricted Beta model.

### 5. Linear approximation of a nonlinear conditional expectation function
The last section studies how the best linear predictor depends on the distribution of the explanatory variable, even when the underlying conditional expectation is quadratic.

## Folder structure

```text
statistics_python_github/
├── data/
│   └── cars.csv
├── statistics_project.py
├── README.md
├── conclusions.md
├── requirements.txt
├── run_output.txt
└── results_summary.json   # optional, created only if SAVE_JSON = True
```

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python statistics_homework.py
```

## What the script produces

When executed, the script:

- prints the main numerical results in the terminal
- explains the purpose of each section before showing the output
- saves the figures into the `figures/` folder
- optionally saves a JSON summary if `SAVE_JSON = True`
- can also display plots directly while running if `SHOW_FIGURES = True`

## Project Value

The value of the project comes from the combination of:

- simulation
- visualization
- regression analysis
- numerical optimization
- asymptotic reasoning
- reproducible scripting

