"""Fit an MOF Casimir-Polder correction factor to published data.

The target is Table I, column (a), for metastable He* near an Au wall at
T = 300 K in J. F. Babb, G. L. Klimchitskaya, and V. M. Mostepanenko,
Phys. Rev. A 70, 042901 (2004), DOI: 10.1103/PhysRevA.70.042901.

The final plasma-wall model fits Omega_A and sqrt(S_W) by nonlinear
least squares. An optional profile scan checks the fixed choice Omega_W = 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.constants import Boltzmann as k_B
from scipy.constants import c, electron_volt, hbar
from scipy.optimize import OptimizeResult, least_squares
from scipy.special import roots_laguerre


# Table I data: metastable He* near Au, column (a), T = 300 K
L_DATA_UM = np.array(
    [
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
        1.50,
        2.00,
        2.50,
        3.00,
        3.50,
        4.00,
        4.50,
        5.00,
        6.00,
        7.00,
        8.00,
    ],
    dtype=float,
)

ETA_DATA = np.array(
    [
        0.5039,
        0.5899,
        0.7070,
        0.7801,
        0.8285,
        0.8620,
        0.8859,
        0.9035,
        0.9167,
        0.9269,
        0.9347,
        0.9409,
        0.9458,
        0.9498,
        0.9531,
        0.9668,
        0.9889,
        1.031,
        1.096,
        1.182,
        1.286,
        1.402,
        1.656,
        1.924,
        2.196,
    ],
    dtype=float,
)

TEMPERATURE_K = 300.0
MATSUBARA_EXPONENT_CUTOFF = 36.0
QUADRATURE_ORDER = 96
INITIAL_OMEGA_A = 1.8e15
INITIAL_WALL_STRENGTH_SQRT = 1.37e16
OMEGA_A_BOUNDS = (1.0e13, 1.0e17)
WALL_STRENGTH_SQRT_BOUNDS = (1.0e14, 1.0e18)

# Numerical quadrature
LAGUERRE_X, LAGUERRE_W = roots_laguerre(QUADRATURE_ORDER)


# Physical model
def validate_data() -> None:
    if L_DATA_UM.shape != ETA_DATA.shape:
        raise ValueError("The separation and correction-factor arrays differ in size.")
    if np.any(L_DATA_UM <= 0.0) or np.any(np.diff(L_DATA_UM) <= 0.0):
        raise ValueError("Separations must be positive and strictly increasing.")
    if np.any(ETA_DATA <= 0.0):
        raise ValueError("Correction factors must be positive.")


def eta_model(
    separation_um: np.ndarray,
    omega_a: float,
    wall_strength_sqrt: float,
    omega_w: float = 0.0,
    temperature: float = TEMPERATURE_K,
) -> np.ndarray:
    """Evaluate the correction factor eta(L) = F(L)/E_ideal(L).

    The atomic response is

        alpha_A(i xi) / alpha_A(0) = Omega_A^2 / (xi^2 + Omega_A^2),

    and the wall response is

        epsilon_W(i xi) = 1 + S_W / (xi^2 + Omega_W^2).

    Frequencies are angular frequencies in rad/s and separations are supplied
    in micrometers. Omega_W = 0 gives the plasma-wall model.
    """

    separation_m = np.atleast_1d(np.asarray(separation_um, dtype=float)) * 1.0e-6
    if np.any(separation_m <= 0.0):
        raise ValueError("All separations must be positive.")
    if omega_a <= 0.0 or wall_strength_sqrt <= 0.0:
        raise ValueError("Omega_A and sqrt(S_W) must be positive.")
    if omega_w < 0.0 or temperature <= 0.0:
        raise ValueError("Omega_W must be nonnegative and temperature positive.")

    eta = np.empty_like(separation_m)
    xi_step = 2.0 * np.pi * k_B * temperature / hbar

    for index, separation in enumerate(separation_m):
        zeta_step = 2.0 * separation * xi_step / c
        n_max = max(
            1, int(np.ceil(MATSUBARA_EXPONENT_CUTOFF / zeta_step))
        )
        matsubara_sum = 0.0

        for n in range(n_max + 1):
            xi_n = n * xi_step
            zeta_n = n * zeta_step
            alpha_ratio = omega_a**2 / (xi_n**2 + omega_a**2)

            if n == 0:
                if omega_w == 0.0:
                    r_tm_zero = 1.0
                else:
                    epsilon_zero = 1.0 + (
                        wall_strength_sqrt / omega_w
                    ) ** 2
                    r_tm_zero = (epsilon_zero - 1.0) / (
                        epsilon_zero + 1.0
                    )
                matsubara_sum += 2.0 * alpha_ratio * r_tm_zero
                continue

            epsilon_w = 1.0 + wall_strength_sqrt**2 / (
                xi_n**2 + omega_w**2
            )
            y = LAGUERRE_X + zeta_n
            wall_root = np.sqrt(
                y**2 + (epsilon_w - 1.0) * zeta_n**2
            )
            r_tm = (epsilon_w * y - wall_root) / (
                epsilon_w * y + wall_root
            )
            r_te = (y - wall_root) / (y + wall_root)
            integrand = (
                (2.0 * y**2 - zeta_n**2) * r_tm
                - zeta_n**2 * r_te
            )
            integral = np.exp(-zeta_n) * np.dot(LAGUERRE_W, integrand)
            matsubara_sum += alpha_ratio * integral

        eta[index] = (
            np.pi
            * k_B
            * temperature
            * separation
            / (3.0 * hbar * c)
            * matsubara_sum
        )

    return eta


# Nonlinear fitting
def fit_at_fixed_omega_w(
    omega_w: float,
    initial_guess: tuple[float, float] = (
        INITIAL_OMEGA_A,
        INITIAL_WALL_STRENGTH_SQRT,
    ),
) -> OptimizeResult:
    """Fit Omega_A and sqrt(S_W) at a fixed value of Omega_W."""

    def residuals(log_parameters: np.ndarray) -> np.ndarray:
        omega_a, wall_strength_sqrt = np.exp(log_parameters)
        return (
            eta_model(
                L_DATA_UM,
                omega_a=omega_a,
                wall_strength_sqrt=wall_strength_sqrt,
                omega_w=omega_w,
            )
            - ETA_DATA
        )

    result = least_squares(
        residuals,
        x0=np.log(initial_guess),
        bounds=(
            np.log([OMEGA_A_BOUNDS[0], WALL_STRENGTH_SQRT_BOUNDS[0]]),
            np.log([OMEGA_A_BOUNDS[1], WALL_STRENGTH_SQRT_BOUNDS[1]]),
        ),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=400,
    )
    if not result.success:
        raise RuntimeError(f"Least-squares fit failed: {result.message}")
    return result


def profile_wall_resonance() -> np.ndarray:
    """Scan Omega_W and refit Omega_A and sqrt(S_W) at each grid point."""

    omega_w_grid = np.concatenate(([0.0], np.logspace(10.0, 16.5, 80)))
    rows = []
    initial_guess = (INITIAL_OMEGA_A, INITIAL_WALL_STRENGTH_SQRT)

    for omega_w in omega_w_grid:
        result = fit_at_fixed_omega_w(omega_w, initial_guess)
        omega_a, wall_strength_sqrt = np.exp(result.x)
        prediction = eta_model(
            L_DATA_UM, omega_a, wall_strength_sqrt, omega_w
        )
        rss = np.sum((prediction - ETA_DATA) ** 2)
        rows.append((omega_w, omega_a, wall_strength_sqrt, rss))
        initial_guess = (omega_a, wall_strength_sqrt)

    return np.asarray(rows)


def calculate_fit_quality(prediction: np.ndarray) -> dict[str, float]:
    residuals = prediction - ETA_DATA
    return {
        "rmse": np.sqrt(np.mean(residuals**2)),
        "max_abs_error": np.max(np.abs(residuals)),
        "r_squared": 1.0
        - np.sum(residuals**2)
        / np.sum((ETA_DATA - ETA_DATA.mean()) ** 2),
    }


# Results and visualization
def save_fit_table(output_path: Path, prediction: np.ndarray) -> None:
    table = np.column_stack(
        (L_DATA_UM, ETA_DATA, prediction, prediction - ETA_DATA)
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="separation_um,eta_table,eta_fit,residual",
        comments="",
        fmt="%.12e",
    )


def save_profile_table(output_path: Path, profile: np.ndarray) -> None:
    np.savetxt(
        output_path,
        profile,
        delimiter=",",
        header="omega_w_rad_s,omega_a_rad_s,wall_strength_sqrt_rad_s,rss",
        comments="",
        fmt="%.12e",
    )


def make_plot(
    omega_a: float,
    wall_strength_sqrt: float,
    omega_w: float,
    output_path: Path,
    show: bool = False,
) -> None:
    """Create a fit plot with a residual panel."""

    import matplotlib.pyplot as plt

    eta_fit = eta_model(
        L_DATA_UM, omega_a, wall_strength_sqrt, omega_w
    )
    dense_separation_um = np.geomspace(
        L_DATA_UM.min(), L_DATA_UM.max(), 260
    )
    dense_eta = eta_model(
        dense_separation_um, omega_a, wall_strength_sqrt, omega_w
    )
    rmse = calculate_fit_quality(eta_fit)["rmse"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
        }
    )
    figure = plt.figure(figsize=(8.4, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[3.5, 1.15])
    axis = figure.add_subplot(grid[0])
    residual_axis = figure.add_subplot(grid[1], sharex=axis)

    axis.plot(dense_separation_um, dense_eta, lw=2.3, label="MOF fit")
    axis.scatter(
        L_DATA_UM,
        ETA_DATA,
        s=34,
        facecolor="white",
        edgecolor="#C62828",
        linewidth=1.5,
        zorder=3,
        label=r"Table I(a), He$^*$ near Au",
    )
    axis.axhline(1.0, color="#6B7280", lw=1.0, ls="--", alpha=0.75)
    axis.set_xscale("log")
    axis.set_ylabel(r"Correction factor $\eta=F/E_{\rm ideal}$")
    axis.set_title("MOF fit to the Casimir-Polder correction factor")
    axis.grid(True, which="major", alpha=0.23)
    axis.grid(True, which="minor", axis="x", alpha=0.10)
    axis.legend(frameon=False, loc="upper left")

    fit_text = (
        rf"$\Omega_A={omega_a / 1e15:.4f}\times10^{{15}}\,\mathrm{{s^{{-1}}}}$"
        + "\n"
        + rf"$\sqrt{{S_W}}={wall_strength_sqrt / 1e16:.4f}"
        + rf"\times10^{{16}}\,\mathrm{{s^{{-1}}}}$"
        + "\n"
        + rf"$\Omega_W={omega_w:.3g}\,\mathrm{{s^{{-1}}}}$; "
        + rf"RMSE $={rmse:.2e}$"
    )
    axis.text(
        0.98,
        0.05,
        fit_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "alpha": 0.94,
        },
    )

    residuals = eta_fit - ETA_DATA
    residual_axis.axhline(0.0, color="#374151", lw=1.0)
    residual_axis.scatter(L_DATA_UM, residuals, s=28)
    residual_axis.set_xscale("log")
    residual_axis.set_xlabel(r"Separation $L$ ($\mu$m)")
    residual_axis.set_ylabel("Fit - table")
    residual_axis.grid(True, which="major", alpha=0.23)
    residual_axis.ticklabel_format(
        axis="y", style="sci", scilimits=(-3, -3)
    )

    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def print_results(
    omega_a: float,
    wall_strength_sqrt: float,
    omega_w: float,
    prediction: np.ndarray,
) -> None:
    quality = calculate_fit_quality(prediction)
    print("Best-fit parameters")
    print(f"Omega_A        = {omega_a:.9e} rad/s")
    print(f"sqrt(S_W)      = {wall_strength_sqrt:.9e} rad/s")
    print(f"S_W            = {wall_strength_sqrt**2:.9e} s^(-2)")
    print(f"Omega_W        = {omega_w:.9e} rad/s")
    print(f"hbar Omega_A   = {hbar * omega_a / electron_volt:.6f} eV")
    print(
        "hbar sqrt(S_W) = "
        f"{hbar * wall_strength_sqrt / electron_volt:.6f} eV"
    )
    print("\nFit quality")
    print(f"RMSE           = {quality['rmse']:.9e}")
    print(f"max abs error  = {quality['max_abs_error']:.9e}")
    print(f"R^2            = {quality['r_squared']:.12f}")
    print("\nL (um)    eta_table    eta_fit       residual")

    for separation, observed, predicted in zip(
        L_DATA_UM, ETA_DATA, prediction
    ):
        print(
            f"{separation:6.2f}    {observed:9.4f}    {predicted:10.6f}"
            f"    {predicted - observed:+.6e}"
        )


# Command-line interface
def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create the PDF plot.",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip the optional Omega_W profile scan.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the fit plot in addition to saving it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mof_fit_output"),
        help="Directory for generated tables and figures.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    validate_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    omega_w_best = 0.0
    result = fit_at_fixed_omega_w(omega_w_best)
    omega_a_best, wall_strength_sqrt_best = np.exp(result.x)
    prediction = eta_model(
        L_DATA_UM,
        omega_a_best,
        wall_strength_sqrt_best,
        omega_w_best,
    )

    print_results(
        omega_a_best,
        wall_strength_sqrt_best,
        omega_w_best,
        prediction,
    )
    save_fit_table(args.output_dir / "mof_eta_fit_results.csv", prediction)

    if not args.no_profile:
        profile = profile_wall_resonance()
        best_row = profile[np.argmin(profile[:, 3])]
        print("\nOmega_W profile optimum")
        print(f"Omega_W        = {best_row[0]:.9e} rad/s")
        print(f"Omega_A        = {best_row[1]:.9e} rad/s")
        print(f"sqrt(S_W)      = {best_row[2]:.9e} rad/s")
        print(f"RSS            = {best_row[3]:.9e}")
        save_profile_table(
            args.output_dir / "omega_w_profile.csv", profile
        )

    if not args.no_plot:
        plot_path = args.output_dir / "mof_eta_fit.pdf"
        make_plot(
            omega_a_best,
            wall_strength_sqrt_best,
            omega_w_best,
            plot_path,
            show=args.show,
        )
        print(f"\nSaved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
