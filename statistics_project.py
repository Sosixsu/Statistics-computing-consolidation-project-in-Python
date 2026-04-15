"""
Statistics Computing Consolidation Project

Each section follows the same sequence:
1. What statistical question is being studied.
2. What the code does.
3. What the main numerical and graphical outcomes are.

Outputs:
- printed results in the terminal
- saved figures in figures/
- optional JSON summary if SAVE_JSON = True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 12345
SHOW_FIGURES = True
SAVE_JSON = False
rng = np.random.default_rng(SEED)
SAVED_FIGURES: list[str] = []


def save_figure(name: str) -> None:
    """Save a figure, optionally display it, and register its path."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    SAVED_FIGURES.append(f"figures/{name}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def to_serializable(value: Any) -> Any:
    """Convert numpy/pandas objects into regular Python types."""
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def print_rule(char: str = "=", width: int = 88) -> None:
    print(char * width)


def print_section(title: str, question: str, purpose: str) -> None:
    print("\n")
    print_rule("=")
    print(title)
    print_rule("=")
    print(f"Question studied: {question}")
    print(f"Purpose: {purpose}")


def print_results_block(title: str, results: dict[str, Any]) -> None:
    print(f"\n{title}")
    print_rule("-")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")
        else:
            print(f"- {key}: {value}")


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Simple OLS implementation with standard errors and confidence intervals."""
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    vcov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(vcov))
    t_stats = beta / se
    crit = stats.t.ppf(0.975, df=n - k)
    ci_low = beta - crit * se
    ci_high = beta + crit * se
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)

    return {
        "beta": beta,
        "fitted": fitted,
        "resid": resid,
        "sigma2": sigma2,
        "se": se,
        "t_stats": t_stats,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r2": r2,
    }


def logistic_neg_loglike(params: np.ndarray, data: np.ndarray) -> float:
    """Negative log-likelihood for a logistic distribution."""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    z = (data - mu) / sigma
    logpdf = -z - np.log(sigma) - 2 * np.log1p(np.exp(-z))
    return -np.sum(logpdf)


def simulate_sample_means(draw_fn, n_values: list[int], simulations: int = 800) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for n in n_values:
        out[n] = np.array([np.mean(draw_fn(n)) for _ in range(simulations)])
    return out


def histogram_grid(sample_dict: dict[int, np.ndarray], title_prefix: str, xlabel: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for ax, (n, values) in zip(axes, sample_dict.items()):
        ax.hist(values, bins=20, edgecolor="black")
        ax.set_title(f"{title_prefix}, N = {n}")
        ax.set_xlabel(xlabel)
    save_figure(filename)


def distribution_transformation_study() -> dict[str, Any]:
    """
    Question:
    Can one common Uniform(0,1) sample be transformed into observations from
    several other distributions without drawing new random numbers?

    Purpose:
    To illustrate inverse-CDF style construction and compare the resulting
    empirical behavior visually.
    """
    print_section(
        "Section 1. Distribution construction from one common random source",
        "Use the same Uniform(0,1) sample to construct observations from several distributions.",
        "Show how one source of randomness can be transformed into different target distributions."
    )

    u = rng.uniform(0, 1, 1000)
    uniform_sample = u.copy()
    logistic_sample = stats.logistic.ppf(u, loc=3, scale=4)
    laplace_sample = stats.laplace.ppf(u, loc=-3, scale=4)
    binomial_sample = stats.binom.ppf(u, n=20, p=0.2).astype(int)
    poisson_sample = stats.poisson.ppf(u, mu=4).astype(int)
    poisson_topcoded = np.minimum(poisson_sample, 10)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    samples = [
        ("Uniform(0,1)", uniform_sample, 30),
        ("Logistic(3,4)", logistic_sample, 30),
        ("Laplace(-3,4)", laplace_sample, 30),
        ("Binomial(20,0.2)", binomial_sample, np.arange(-0.5, 21.5, 1)),
        ("Poisson(4), top-coded at 10", poisson_topcoded, np.arange(-0.5, 11.5, 1)),
    ]
    axes = axes.flatten()
    for ax, (title, sample, bins) in zip(axes, samples):
        ax.hist(sample, bins=bins, edgecolor="black")
        ax.set_title(title)
    axes[-1].axis("off")
    save_figure("distribution_transformations.png")

    results = {
        "uniform_mean": np.mean(uniform_sample),
        "logistic_mean": np.mean(logistic_sample),
        "laplace_mean": np.mean(laplace_sample),
        "binomial_mean": np.mean(binomial_sample),
        "poisson_topcoded_mean": np.mean(poisson_topcoded),
    }
    print_results_block("Outcome", to_serializable(results))
    return to_serializable(results)


def multivariate_dependence_study() -> dict[str, Any]:
    """
    Question:
    What does dependence look like in simulated bivariate data, and how do
    conditional means and grouped empirical calculations respond to the sign
    of correlation?

    Purpose:
    To study dependence, quantile conditioning, and empirical expectation/
    variance decompositions in simulated data.
    """
    print_section(
        "Section 2. Dependence, conditioning, and grouped empirical analysis",
        "Simulate bivariate data, split one variable into deciles, study conditional means, and compare positive and negative dependence.",
        "Show how dependence affects scatter patterns, conditional averages, and grouped variance calculations."
    )

    mean = np.array([2.0, 5.0])
    cov = np.array([[2.0, 0.4 * np.sqrt(2.0) * np.sqrt(4.0)],
                    [0.4 * np.sqrt(2.0) * np.sqrt(4.0), 4.0]])
    data = rng.multivariate_normal(mean=mean, cov=cov, size=1000)
    x1, x2 = data[:, 0], data[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].hist(x1, bins=30, edgecolor="black")
    axes[0, 0].set_title("x1")
    axes[0, 1].scatter(x1, x2, s=12, alpha=0.6)
    axes[0, 1].set_title("Scatter: x1 vs x2")
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")
    axes[1, 0].axis("off")
    axes[1, 1].hist(x2, bins=30, edgecolor="black")
    axes[1, 1].set_title("x2")
    save_figure("multivariate_normal_overview.png")

    q40 = np.quantile(x1, 0.4)
    q80 = np.quantile(x1, 0.8)
    mean_x2_above_40 = np.mean(x2[x1 > q40])
    mean_x2_40_80 = np.mean(x2[(x1 > q40) & (x1 < q80)])

    decile_edges = np.quantile(x1, np.linspace(0, 1, 11))
    decile_edges[0] -= 1e-9
    decile_groups = pd.cut(x1, bins=decile_edges, labels=False, include_lowest=True) + 1
    df = pd.DataFrame({"x1": x1, "x2": x2, "decile": decile_groups})

    empirical_mean_x1 = df["x1"].mean()
    empirical_mean_from_groups = (
        df.groupby("decile")["x1"].mean().mul(df.groupby("decile").size() / len(df)).sum()
    )

    within = (df.groupby("decile")["x1"].var(ddof=1) * (df.groupby("decile").size() / len(df))).sum()
    between = df.groupby("decile")["x1"].mean().var(ddof=1)
    total = df["x1"].var(ddof=1)

    cov_neg = np.array([[2.0, -0.4 * np.sqrt(2.0) * np.sqrt(4.0)],
                        [-0.4 * np.sqrt(2.0) * np.sqrt(4.0), 4.0]])
    data_neg = rng.multivariate_normal(mean=mean, cov=cov_neg, size=1000)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data_neg[:, 0], data_neg[:, 1], s=12, alpha=0.6)
    ax.set_title("Bivariate normal with negative correlation")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    save_figure("multivariate_negative_correlation.png")

    lognormal_data = np.exp(data)
    lx1, lx2 = lognormal_data[:, 0], lognormal_data[:, 1]
    lq40 = np.quantile(lx1, 0.4)
    lq80 = np.quantile(lx1, 0.8)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(lx1, lx2, s=12, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Lognormal version of the simulated data")
    ax.set_xlabel("exp(x1)")
    ax.set_ylabel("exp(x2)")
    save_figure("multivariate_lognormal_view.png")

    results = {
        "mean_x2_given_x1_above_q40": mean_x2_above_40,
        "mean_x2_given_q40_lt_x1_lt_q80": mean_x2_40_80,
        "empirical_mean_x1": empirical_mean_x1,
        "empirical_mean_x1_from_groups": empirical_mean_from_groups,
        "total_variance_x1": total,
        "within_group_component": within,
        "between_group_component": between,
        "correlation_original": np.corrcoef(x1, x2)[0, 1],
        "correlation_negative_case": np.corrcoef(data_neg[:, 0], data_neg[:, 1])[0, 1],
        "lognormal_mean_x2_given_x1_above_q40": np.mean(lx2[lx1 > lq40]),
        "lognormal_mean_x2_given_q40_lt_x1_lt_q80": np.mean(lx2[(lx1 > lq40) & (lx1 < lq80)]),
    }
    print_results_block("Outcome", to_serializable(results))
    return to_serializable(results)


def regression_and_mle_study() -> dict[str, Any]:
    """
    Question:
    What does a simple linear regression show in the cars dataset, and can
    numerical MLE recover logistic distribution parameters from a simulated sample?

    Purpose:
    To combine descriptive statistics, regression analysis, residual diagnostics,
    simulation-based coefficient stability, and numerical maximum likelihood.
    """
    print_section(
        "Section 3. Regression analysis, diagnostics, and numerical maximum likelihood",
        "Study a simple regression with the cars data, inspect residual behavior, simulate related regressions, and estimate logistic parameters by MLE.",
        "Show how descriptive analysis, OLS, diagnostics, simulation, and numerical optimization fit together in one workflow."
    )

    cars = pd.read_csv(DATA_DIR / "C:/Users/User/Downloads/statistics_python_github/data/cars.csv")
    x = cars["speed"].to_numpy()
    y = cars["dist"].to_numpy()

    summary = {
        "sample_size": int(len(cars)),
        "speed_mean": np.mean(x),
        "speed_variance": np.var(x, ddof=1),
        "distance_mean": np.mean(y),
        "distance_variance": np.var(y, ddof=1),
        "speed_quantiles": cars["speed"].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
        "distance_quantiles": cars["dist"].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
    }

    fit = ols_fit(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=28)
    x_grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_grid, fit["beta"][0] + fit["beta"][1] * x_grid)
    ax.set_title("Cars dataset: distance on speed")
    ax.set_xlabel("speed")
    ax.set_ylabel("distance")
    save_figure("cars_scatter_and_regression.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, fit["resid"], s=28)
    ax.axhline(0, linestyle="--")
    ax.set_title("Regression residuals")
    ax.set_xlabel("speed")
    ax.set_ylabel("residual")
    save_figure("cars_residual_plot.png")

    residual_sd = np.std(fit["resid"], ddof=1)
    newvar1 = fit["fitted"] + rng.normal(0, residual_sd, size=len(x))
    newvar2 = x + rng.normal(0, np.std(x, ddof=1), size=len(x))
    fit_new = ols_fit(newvar2, newvar1)

    repeated_slopes = []
    repeated_r2 = []
    for _ in range(10):
        y_sim = fit["fitted"] + rng.normal(0, residual_sd, size=len(x))
        x_sim = x + rng.normal(0, np.std(x, ddof=1), size=len(x))
        sim_fit = ols_fit(x_sim, y_sim)
        repeated_slopes.append(sim_fit["beta"][1])
        repeated_r2.append(sim_fit["r2"])

    true_mu, true_sigma = 1.0, 3.0
    logistic_sample = rng.logistic(loc=true_mu, scale=true_sigma, size=1000)
    initial = np.array([
        np.median(logistic_sample),
        np.std(logistic_sample, ddof=1) * np.sqrt(3) / np.pi
    ])

    opt = optimize.minimize(
        logistic_neg_loglike,
        x0=initial,
        args=(logistic_sample,),
        method="L-BFGS-B",
        bounds=[(None, None), (1e-8, None)],
    )

    results = {
        "descriptive_summary": summary,
        "ols_intercept": fit["beta"][0],
        "ols_slope": fit["beta"][1],
        "ols_slope_standard_error": fit["se"][1],
        "ols_slope_t_statistic": fit["t_stats"][1],
        "ols_slope_confidence_interval_95": [fit["ci_low"][1], fit["ci_high"][1]],
        "ols_r_squared": fit["r2"],
        "simulated_regression_intercept": fit_new["beta"][0],
        "simulated_regression_slope": fit_new["beta"][1],
        "simulated_regression_r_squared": fit_new["r2"],
        "repeated_simulation_slope_mean": np.mean(repeated_slopes),
        "repeated_simulation_slope_std": np.std(repeated_slopes, ddof=1),
        "repeated_simulation_r2_mean": np.mean(repeated_r2),
        "logistic_true_mu": true_mu,
        "logistic_true_sigma": true_sigma,
        "logistic_mle_mu": opt.x[0],
        "logistic_mle_sigma": opt.x[1],
        "logistic_optimizer_success": bool(opt.success),
    }
    print_results_block("Outcome", to_serializable(results))
    return to_serializable(results)


def asymptotics_and_predictors_study() -> dict[str, Any]:
    """
    Question:
    How do sample averages behave under repeated simulation, how do MM and MLE
    compare in a restricted Beta model, and how does the quality of a linear
    predictor depend on the distribution of X?

    Purpose:
    To combine asymptotic simulation, estimator comparison, and approximation
    analysis in one final section.
    """
    print_section(
        "Section 4. Asymptotics, estimator comparison, and linear approximation",
        "Use repeated simulation to study LLN and CLT behavior, compare MM and MLE in a Beta model, and examine the best linear predictor under different distributions of X.",
        "Show how large-sample behavior, estimator efficiency, and approximation quality can all be investigated computationally."
    )

    n_values = [1, 10, 100, 1000]

    poisson_means = simulate_sample_means(lambda n: rng.poisson(4, size=n), n_values)
    histogram_grid(poisson_means, "Poisson sample means", "sample mean", "poisson_lln.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    clt_diagnostics = {}
    for ax, n in zip(axes, n_values):
        means = poisson_means[n]
        z = np.sqrt(n) * (means - 4) / np.sqrt(4)
        ax.hist(z, bins=20, density=True, edgecolor="black")
        grid = np.linspace(z.min(), z.max(), 300)
        ax.plot(grid, stats.norm.pdf(grid))
        ax.set_title(f"Poisson CLT, N = {n}")
        ax.set_xlabel("standardized mean")
        clt_diagnostics[str(n)] = {
            "z_mean": np.mean(z),
            "z_std": np.std(z, ddof=1),
        }
    save_figure("poisson_clt.png")

    exp_means = simulate_sample_means(lambda n: rng.exponential(scale=4, size=n), n_values)
    histogram_grid(exp_means, "Exponential sample means", "sample mean", "exponential_lln.png")

    cauchy_medians = {
        n: np.array([np.median(rng.standard_cauchy(size=n)) for _ in range(800)])
        for n in n_values
    }
    histogram_grid(cauchy_medians, "Cauchy sample medians", "sample median", "cauchy_medians.png")

    def inid_draw(n: int) -> np.ndarray:
        means = np.linspace(-2, 2, n)
        return rng.normal(loc=means, scale=1.0, size=n) - means

    inid_means = simulate_sample_means(inid_draw, n_values)
    histogram_grid(inid_means, "i.n.i.d. adjusted means", "sample mean", "inid_means.png")

    beta_true = 2.0
    alpha = 1.0
    num_simulations = 1000
    sample_size = 100
    mm_est = []
    mle_est = []

    def neg_loglike_beta(beta_param: np.ndarray, sample: np.ndarray) -> float:
        beta_value = beta_param[0]
        if beta_value <= 0:
            return np.inf
        return -np.sum(stats.beta.logpdf(sample, a=alpha, b=beta_value))

    for _ in range(num_simulations):
        sample = rng.beta(alpha, beta_true, size=sample_size)
        mm = 1 / np.mean(sample) - 1
        mm_est.append(mm)

        init = np.array([max(mm, 1e-6)])
        opt = optimize.minimize(
            neg_loglike_beta,
            x0=init,
            args=(sample,),
            method="L-BFGS-B",
            bounds=[(1e-8, None)],
        )
        mle_est.append(float(opt.x[0]))

    mm_est = np.array(mm_est)
    mle_est = np.array(mle_est)

    def cef(x, a=0.0, b=1.0, c=-0.1):
        return a + b * x + c * x**2

    x_uniform = rng.uniform(0, 1, 1000)
    y_uniform = cef(x_uniform)
    fit_uniform = ols_fit(x_uniform, y_uniform)

    x_beta = rng.beta(2, 5, 1000)
    y_beta = cef(x_beta)
    fit_beta = ols_fit(x_beta, y_beta)

    x_normal_small = rng.normal(0, 1, 1000)
    y_normal_small = cef(x_normal_small)
    fit_normal_small = ols_fit(x_normal_small, y_normal_small)

    x_normal_large = rng.normal(0, 5, 1000)
    y_normal_large = cef(x_normal_large)
    fit_normal_large = ols_fit(x_normal_large, y_normal_large)

    x_exponential = rng.exponential(scale=np.sqrt(10), size=1000)
    y_exponential = cef(x_exponential)
    fit_exponential = ols_fit(x_exponential, y_exponential)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_specs = [
        (axes[0, 0], x_uniform, y_uniform, fit_uniform["beta"], "Uniform X"),
        (axes[0, 1], x_beta, y_beta, fit_beta["beta"], "Beta X"),
        (axes[1, 0], x_normal_small, y_normal_small, fit_normal_small["beta"], "Normal X, small variance"),
        (axes[1, 1], x_exponential, y_exponential, fit_exponential["beta"], "Exponential X"),
    ]
    for ax, xvals, yvals, beta, title in plot_specs:
        ax.scatter(xvals, yvals, s=10, alpha=0.4)
        grid = np.linspace(np.min(xvals), np.max(xvals), 300)
        ax.plot(grid, cef(grid), linewidth=2, label="quadratic CEF")
        ax.plot(grid, beta[0] + beta[1] * grid, linewidth=2, label="linear predictor")
        ax.set_title(title)
        ax.legend()
    save_figure("linear_predictor_vs_quadratic_cef.png")

    results = {
        "poisson_clt_diagnostics": clt_diagnostics,
        "beta_mm_mean": np.mean(mm_est),
        "beta_mm_variance": np.var(mm_est, ddof=1),
        "beta_mm_mse": np.mean((mm_est - beta_true) ** 2),
        "beta_mle_mean": np.mean(mle_est),
        "beta_mle_variance": np.var(mle_est, ddof=1),
        "beta_mle_mse": np.mean((mle_est - beta_true) ** 2),
        "linear_predictor_coefficients_uniform_x": fit_uniform["beta"],
        "linear_predictor_coefficients_beta_x": fit_beta["beta"],
        "linear_predictor_coefficients_normal_small_var": fit_normal_small["beta"],
        "linear_predictor_coefficients_normal_large_var": fit_normal_large["beta"],
        "linear_predictor_coefficients_exponential_x": fit_exponential["beta"],
    }
    print_results_block("Outcome", to_serializable(results))
    return to_serializable(results)


def main() -> None:
    print_rule("=")
    print("STATISTICS COMPUTING CONSOLIDATION PROJECT")
    print_rule("=")
    print("This run will compute all sections, print the results, and save the figures.")
    print(f"Display figures while running: {SHOW_FIGURES}")
    print(f"Save JSON summary: {SAVE_JSON}")

    results = {
        "distribution_construction": distribution_transformation_study(),
        "multivariate_dependence": multivariate_dependence_study(),
        "regression_and_mle": regression_and_mle_study(),
        "asymptotics_and_linear_predictors": asymptotics_and_predictors_study(),
    }

    if SAVE_JSON:
        with open(ROOT / "results_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\n")
    print_rule("=")
    print("GENERATED FIGURES")
    print_rule("=")
    for path in SAVED_FIGURES:
        print(f"- {path}")

    print("\n")
    print_rule("=")
    print("FINAL TAKEAWAY")
    print_rule("=")
    print(
        "This project is best understood as a statistics computing consolidation project: "
        "it combines simulation, dependence analysis, regression, maximum likelihood, "
        "asymptotic reasoning, and visualization in one reproducible Python workflow."
    )


if __name__ == "__main__":
    main()
