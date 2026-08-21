"""Fit a single-oscillator model to tabulated atomic polarizabilities.

The tabulated data are from Table A of A. Derevianko, S. G. Porsev, and
J. F. Babb, Atomic Data and Nuclear Data Tables 96, 323--331 (2010).
All fitted quantities are expressed in Hartree atomic units unless an SI unit
is shown explicitly.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


# Physical constants and model convention
OMEGA_P_SI = 1.2e16
AU_ANGULAR_FREQUENCY_SI = 4.1341373336493e16
ELECTRON_MASS_AU = 1.0

# Imaginary-frequency grid from Derevianko, Porsev, and Babb (2010)
xi_au = np.array([
    0.0,
    0.00178065, 0.00937463, 0.0230068, 0.042631, 0.0681817,
    0.0995817, 0.136748, 0.179602, 0.228071, 0.282107,
    0.341686, 0.406823, 0.477579, 0.554072, 0.636488,
    0.725091, 0.820235, 0.922382, 1.03212, 1.15017,
    1.27743, 1.41501, 1.56425, 1.72679, 1.90461,
    2.10016, 2.31644, 2.55713, 2.82683, 3.13128,
    3.47776, 3.87553, 4.3366, 4.87665, 5.51655,
    6.28448, 7.21927, 8.37558, 9.83228, 11.7066,
    14.179, 17.5384, 22.2715, 29.2508, 40.168,
    58.6667, 93.8285, 173.862, 426.684, 2246.37
])

# Dynamic polarizability data in atomic units
alpha_data_au = {
    "He": np.array([
        1.383,
        1.3832, 1.3831, 1.3824, 1.3804, 1.3761,
        1.3681, 1.355, 1.3355, 1.3081, 1.2721,
        1.227, 1.1732, 1.1117, 1.0439, 9.7176e-1,
        8.9716e-1, 8.2198e-1, 7.4784e-1, 6.7605e-1,
        6.0757e-1, 5.4303e-1, 4.8284e-1, 4.2714e-1,
        3.7596e-1, 3.2919e-1, 2.8667e-1, 2.4818e-1,
        2.135e-1, 1.8237e-1, 1.5457e-1, 1.2987e-1,
        1.0804e-1, 8.8889e-2, 7.2211e-2, 5.7821e-2,
        4.5539e-2, 3.5189e-2, 2.6598e-2, 1.9593e-2,
        1.4001e-2, 9.648e-3, 6.3624e-3, 3.9735e-3,
        2.316e-3, 1.2329e-3, 5.795e-4, 2.2691e-4,
        6.614e-5, 1.0985e-5, 3.9634e-7
    ]),

    "Be": np.array([
        37.76,
        3.7761e1, 3.7678e1, 3.7253e1, 3.6068e1, 3.3715e1,
        3.008e1, 2.5525e1, 2.0729e1, 1.6321e1, 1.2635e1,
        9.7285, 7.5094, 5.8375, 4.5801, 3.6298,
        2.9053, 2.3471, 1.9124, 1.57, 1.2974,
        1.0782, 9.0039e-1, 7.548e-1, 6.3468e-1,
        5.3487e-1, 4.5139e-1, 3.8116e-1, 3.2178e-1,
        2.7133e-1, 2.2829e-1, 1.9145e-1, 1.5981e-1,
        1.3258e-1, 1.0911e-1, 8.888e-2, 7.1473e-2,
        5.6557e-2, 4.3871e-2, 3.3206e-2, 2.4388e-2,
        1.7262e-2, 1.1673e-2, 7.4569e-3, 4.4315e-3,
        2.3965e-3, 1.1397e-3, 4.4978e-4, 1.3169e-4,
        2.1915e-5, 7.9107e-7
    ]),

    "Ar": np.array([
        11.08,
        1.1082e1, 1.1079e1, 1.1067e1, 1.1031e1, 1.0954e1,
        1.0814e1, 1.0591e1, 1.0269e1, 9.8423, 9.3171,
        8.7097, 8.044, 7.3471, 6.6443, 5.9571,
        5.3018, 4.6893, 4.1258, 3.6139, 3.1534,
        2.7424, 2.3777, 2.0555, 1.7719, 1.5229,
        1.3049, 1.1142, 9.4788e-1, 8.0296e-1,
        6.7699e-1, 5.6773e-1, 4.7325e-1, 3.9181e-1,
        3.2189e-1, 2.6211e-1, 2.1123e-1, 1.6814e-1,
        1.3181e-1, 1.0136e-1, 7.6035e-2, 5.5247e-2,
        3.8541e-2, 2.5547e-2, 1.5877e-2, 9.0623e-3,
        4.5722e-3, 1.9e-3, 5.7452e-4, 9.7061e-5,
        3.5573e-6
    ]),

    "Xe": np.array([
        27.16,
        2.7156e1, 2.7145e1, 2.7088e1, 2.6923e1, 2.6571e1,
        2.5947e1, 2.4986e1, 2.367e1, 2.2034e1, 2.0164e1,
        1.8172e1, 1.6165e1, 1.4233e1, 1.2437e1, 1.0809e1,
        9.3621, 8.0938, 6.9925, 6.0419, 5.2242,
        4.5216, 3.9179, 3.3983, 2.9499, 2.5615,
        2.2239, 1.9291, 1.6705, 1.4427, 1.2414,
        1.0627, 9.0391e-1, 7.6262e-1, 6.3702e-1,
        5.2567e-1, 4.2747e-1, 3.4156e-1, 2.673e-1,
        2.0416e-1, 1.5163e-1, 1.0912e-1, 7.5839e-2,
        5.0763e-2, 3.2591e-2, 1.9829e-2, 1.1001e-2,
        5.073e-3, 1.6545e-3, 2.8817e-4, 1.0532e-5
    ]),

    "Li": np.array([
        164.0,
        1.6386e2, 1.6094e2, 1.4727e2, 1.181e2, 8.2483e1,
        5.3094e1, 3.355e1, 2.1584e1, 1.4337e1, 9.8574,
        7.0022, 5.1216, 3.844, 2.9512, 2.3111,
        1.8417, 1.4902, 1.2221, 1.0139, 8.4974e-1,
        7.1822e-1, 6.1136e-1, 5.2334e-1, 4.4989e-1,
        3.8785e-1, 3.3485e-1, 2.891e-1, 2.4925e-1,
        2.1428e-1, 1.834e-1, 1.5602e-1, 1.3171e-1,
        1.1013e-1, 9.1031e-2, 7.4209e-2, 5.9517e-2,
        4.6826e-2, 3.6019e-2, 2.6979e-2, 1.9581e-2,
        1.3686e-2, 9.1389e-3, 5.7691e-3, 3.3923e-3,
        1.8183e-3, 8.5884e-4, 3.3735e-4, 9.8504e-5,
        1.6372e-5, 5.9081e-7
    ]),

    "Cs": np.array([
        399.8,
        3.9938e2, 3.88e2, 3.3819e2, 2.4771e2, 1.5921e2,
        9.9455e1, 6.4749e1, 4.5037e1, 3.343e1, 2.6166e1,
        2.1283e1, 1.776e1, 1.5058e1, 1.2886e1, 1.1086e1,
        9.5655, 8.2676, 7.1542, 6.1966, 5.372,
        4.6611, 4.0476, 3.5171, 3.0576, 2.6583,
        2.3104, 2.0061, 1.739, 1.5036, 1.2953,
        1.1104, 9.4575e-1, 7.9902e-1, 6.683e-1,
        5.5221e-1, 4.4973e-1, 3.6009e-1, 2.827e-1,
        2.1698e-1, 1.6227e-1, 1.1779e-1, 8.2549e-2,
        5.5437e-2, 3.5233e-2, 2.0733e-2, 1.0867e-2,
        4.7421e-3, 1.5266e-3, 2.7665e-4, 1.0596e-5
    ])
}

# Model and fitting
def alpha_model_imag(xi, lam, omega):
    return lam**2 / (ELECTRON_MASS_AU * (xi**2 + omega**2))


def validate_data():
    for element, alpha_au in alpha_data_au.items():
        if alpha_au.shape != xi_au.shape:
            raise ValueError(
                f"{element} has {alpha_au.size} polarizabilities, "
                f"but the frequency grid has {xi_au.size} points."
            )
        if np.any(alpha_au <= 0.0):
            raise ValueError(f"{element} contains a nonpositive polarizability.")


def fit_element(element, use_log_fit=True, omega_ratio_guess=0.7):
    alpha_au = alpha_data_au[element]
    omega_guess_au = (
        omega_ratio_guess * OMEGA_P_SI / AU_ANGULAR_FREQUENCY_SI
    )
    lambda_guess = omega_guess_au * np.sqrt(
        ELECTRON_MASS_AU * alpha_au[0]
    )

    def residuals(parameters):
        lam, omega = parameters
        alpha_pred = alpha_model_imag(xi_au, lam, omega)
        if use_log_fit:
            return np.log(alpha_pred) - np.log(alpha_au)
        return alpha_pred - alpha_au

    result = least_squares(
        residuals,
        x0=np.array([lambda_guess, omega_guess_au]),
        bounds=(np.full(2, 1.0e-15), np.full(2, np.inf)),
        x_scale="jac",
    )
    if not result.success:
        raise RuntimeError(f"Fit failed for {element}: {result.message}")

    lambda_fit, omega_fit_au = result.x
    alpha_pred = alpha_model_imag(xi_au, lambda_fit, omega_fit_au)
    residual_abs = alpha_pred - alpha_au
    residual_rel = residual_abs / alpha_au
    residual_log = np.log(alpha_pred) - np.log(alpha_au)
    omega_fit_si = omega_fit_au * AU_ANGULAR_FREQUENCY_SI

    return {
        "element": element,
        "lambda": lambda_fit,
        "lambda2": lambda_fit**2,
        "omega_au": omega_fit_au,
        "omega_si": omega_fit_si,
        "r_p": omega_fit_si / OMEGA_P_SI,
        "rmse": np.sqrt(np.mean(residual_abs**2)),
        "relative_rmse": np.sqrt(np.mean(residual_rel**2)),
        "log_rmse": np.sqrt(np.mean(residual_log**2)),
        "r_squared": 1.0 - np.sum(residual_abs**2) / np.sum(
            (alpha_au - np.mean(alpha_au))**2
        ),
        "alpha_data": alpha_au,
        "alpha_pred": alpha_pred,
        "relative_residual": residual_rel,
    }


# Results table and figures
def print_summary(fits, element_order):
    columns = (
        f"{'El.':<4} {'lambda':>14} {'lambda^2':>14} "
        f"{'Omega [a.u.]':>14} {'Omega [s^-1]':>16} {'r_p':>10} "
        f"{'RMSE [a.u.]':>14} {'rel. RMSE':>12} {'log RMSE':>12} "
        f"{'R^2':>10}"
    )
    print("\nFit results (m = m_e = 1 a.u.)")
    print("-" * len(columns))
    print(columns)
    print("-" * len(columns))

    for element in element_order:
        fit = fits[element]
        print(
            f"{element:<4} {fit['lambda']:14.6e} {fit['lambda2']:14.6e} "
            f"{fit['omega_au']:14.6e} {fit['omega_si']:16.6e} "
            f"{fit['r_p']:10.6f} {fit['rmse']:14.6e} "
            f"{fit['relative_rmse']:12.6e} {fit['log_rmse']:12.6e} "
            f"{fit['r_squared']:10.6f}"
        )


def save_summary_csv(fits, element_order, output_path):
    fieldnames = [
        "element",
        "lambda_au",
        "lambda_squared_au",
        "omega_au",
        "omega_si_s^-1",
        "omega_over_omega_p",
        "rmse_au",
        "relative_rmse",
        "log_rmse",
        "r_squared",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for element in element_order:
            fit = fits[element]
            writer.writerow(
                {
                    "element": element,
                    "lambda_au": fit["lambda"],
                    "lambda_squared_au": fit["lambda2"],
                    "omega_au": fit["omega_au"],
                    "omega_si_s^-1": fit["omega_si"],
                    "omega_over_omega_p": fit["r_p"],
                    "rmse_au": fit["rmse"],
                    "relative_rmse": fit["relative_rmse"],
                    "log_rmse": fit["log_rmse"],
                    "r_squared": fit["r_squared"],
                }
            )


def plot_fits(fits, element_order, output_path, show=False):
    positive_frequency = xi_au > 0.0
    xi_smooth = np.geomspace(
        xi_au[positive_frequency].min(),
        xi_au[positive_frequency].max(),
        1000,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for axis, element in zip(axes.ravel(), element_order):
        fit = fits[element]
        alpha_smooth = alpha_model_imag(
            xi_smooth, fit["lambda"], fit["omega_au"]
        )
        axis.loglog(
            xi_au[positive_frequency],
            fit["alpha_data"][positive_frequency],
            "o",
            markersize=4,
            alpha=0.7,
            label=f"{element} data",
        )
        axis.loglog(
            xi_smooth,
            alpha_smooth,
            "-",
            linewidth=2,
            label=f"{element} fit",
        )
        axis.set_title(
            rf"{element}: $\lambda={fit['lambda']:.3e}$, "
            rf"$\Omega={fit['omega_au']:.3e}$ a.u."
        )
        axis.set_xlabel(r"Imaginary frequency $\xi$ [a.u.]")
        axis.set_ylabel(r"$\alpha(i\xi)$ [a.u.]")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=9)

    fig.suptitle(
        r"Two-parameter fits with $m=m_e$: "
        r"$\alpha(i\xi)=\lambda^2/[m_e(\xi^2+\Omega^2)]$",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_relative_residuals(fits, element_order, output_path, show=False):
    positive_frequency = xi_au > 0.0
    fig, axis = plt.subplots(figsize=(10, 6))

    for element in element_order:
        axis.semilogx(
            xi_au[positive_frequency],
            fits[element]["relative_residual"][positive_frequency],
            "o-",
            markersize=4,
            linewidth=1.5,
            label=element,
        )

    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel(r"Imaginary frequency $\xi$ [a.u.]")
    axis.set_ylabel(
        r"$(\alpha_{\mathrm{model}}-\alpha_{\mathrm{data}})"
        r"/\alpha_{\mathrm{data}}$"
    )
    axis.set_title("Relative residuals for all fitted elements")
    axis.legend(ncols=2)
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# Command-line interface
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-scale",
        choices=("log", "linear"),
        default="log",
        help="Residual scale used by least-squares fitting (default: log).",
    )
    parser.add_argument(
        "--omega-ratio-guess",
        type=float,
        default=0.7,
        help="Initial guess Omega/Omega_P (default: 0.7).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fit_output"),
        help="Directory for the CSV table and figures (default: fit_output).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    element_order = ["Xe", "He", "Ar", "Be", "Li", "Cs"]
    validate_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fits = {
        element: fit_element(
            element,
            use_log_fit=args.fit_scale == "log",
            omega_ratio_guess=args.omega_ratio_guess,
        )
        for element in element_order
    }

    print_summary(fits, element_order)
    save_summary_csv(
        fits, element_order, args.output_dir / "polarizability_fit_results.csv"
    )
    plot_fits(
        fits,
        element_order,
        args.output_dir / "polarizability_fits.pdf",
        show=args.show,
    )
    plot_relative_residuals(
        fits,
        element_order,
        args.output_dir / "polarizability_relative_residuals.pdf",
        show=args.show,
    )


if __name__ == "__main__":
    main()
