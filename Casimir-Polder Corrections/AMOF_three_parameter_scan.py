#!/usr/bin/env python3
"""Simultaneous, multi-start 3-parameter AMOF Casimir-Polder fit.

Run from the directory containing the original mof_free_energy_fit.py:
    python AMOF_three_parameter_scan.py

The original eta_model(), tabulated data, integration and sum settings are
reused unchanged. Fit parameters are Omega_A, Omega_W, sqrt(S_W).
For conditioning, optimize (log(Omega_A / 1e15),
log(sqrt(S_W) / 1e16), (Omega_W / 1e15)^2). Parameterizing Omega_W^2
allows the optimizer to reach the plasma boundary Omega_W = 0.

Produces AMOF_three_parameter_starts.csv and
AMOF_three_parameter_predictions.csv.
"""
import csv
import itertools
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
import mof_free_energy_fit as model

A_SCALE = 1e15
S_SCALE = 1e16
W_SCALE = 1e15
W_MAX = 10**16.5  # same positive limit as existing fixed-W profile scan


def encode(a, s, w):
    return np.array([np.log(a / A_SCALE), np.log(s / S_SCALE), (w / W_SCALE)**2])


def decode(z):
    return A_SCALE * np.exp(z[0]), S_SCALE * np.exp(z[1]), W_SCALE * np.sqrt(max(z[2], 0))


LOWER = np.array([
    np.log(model.OMEGA_A_BOUNDS[0] / A_SCALE),
    np.log(model.WALL_STRENGTH_SQRT_BOUNDS[0] / S_SCALE),
    0.0,
])
UPPER = np.array([
    np.log(model.OMEGA_A_BOUNDS[1] / A_SCALE),
    np.log(model.WALL_STRENGTH_SQRT_BOUNDS[1] / S_SCALE),
    (W_MAX / W_SCALE)**2,
])


def residual(z):
    a, s, w = decode(z)
    return model.eta_model(model.L_DATA_UM, a, s, omega_w=w) - model.ETA_DATA


def fit(start):
    return least_squares(residual, encode(*start), bounds=(LOWER, UPPER),
                         method='trf', x_scale='jac',
                         xtol=1e-12, ftol=1e-12, gtol=1e-12,
                         max_nfev=300)


def main():
    model.validate_data()
    # Physically motivated and relatively distant initial guesses.
    a_starts = [0.8e15, 1.8e15, 3.0e15]
    s_starts = [0.8e16, 1.37e16, 3.0e16]
    w_starts = [0.0, 1.0e13, 1.0e14, 1.0e15, 1.0e16]
    starts = list(itertools.product(a_starts, s_starts, w_starts))
    records = []
    best = None
    start_time = time.time()
    for i, start in enumerate(starts, 1):
        result = fit(start)
        a, s, w = decode(result.x)
        rss = float(np.sum(result.fun**2))
        row = dict(initial_Omega_A=start[0], initial_sqrt_S_W=start[1],
                   initial_Omega_W=start[2], fitted_Omega_A=a,
                   fitted_sqrt_S_W=s, fitted_Omega_W=w,
                   fitted_S_W=s**2, RSS=rss,
                   RMSE=np.sqrt(rss / model.L_DATA_UM.size),
                   max_abs_residual=np.max(np.abs(result.fun)),
                   nfev=result.nfev, success=result.success,
                   termination=result.message)
        records.append(row)
        if result.success and (best is None or rss < best[0]):
            best = (rss, row, result)
        if i % 10 == 0 or i == len(starts):
            print(f'{i}/{len(starts)} fits; elapsed {time.time()-start_time:.1f}s; '
                  f'best RSS {best[0]:.12g}', flush=True)

    output_dir = Path(__file__).resolve().parent
    with (output_dir / 'AMOF_three_parameter_starts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Compare with original two-parameter plasma-wall fit.
    plasma_result = model.fit_at_fixed_omega_w(0.0)
    a_plasma, s_plasma = np.exp(plasma_result.x)
    r_plasma = model.eta_model(model.L_DATA_UM, a_plasma, s_plasma, 0.0) - model.ETA_DATA
    plasma_rss = np.sum(r_plasma**2)

    row = best[1]
    predictions = model.eta_model(model.L_DATA_UM, row['fitted_Omega_A'],
                                  row['fitted_sqrt_S_W'], row['fitted_Omega_W'])
    with (output_dir / 'AMOF_three_parameter_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['L_um','eta_reference','eta_three_param','residual_three_param',
                         'eta_plasma_fit','residual_plasma_fit'])
        for distance, target, result_eta, pres in zip(model.L_DATA_UM, model.ETA_DATA, predictions, r_plasma):
            writer.writerow([distance,target,result_eta,result_eta-target,target+pres,pres])

    print('\nBest three-parameter result')
    print(f'Omega_A           = {row["fitted_Omega_A"]:.12e} rad/s')
    print(f'Omega_W           = {row["fitted_Omega_W"]:.12e} rad/s')
    print(f'sqrt(S_W)         = {row["fitted_sqrt_S_W"]:.12e} rad/s')
    print(f'S_W               = {row["fitted_S_W"]:.12e} rad^2/s^2')
    print(f'RSS               = {best[0]:.12e}')
    print(f'RMSE              = {row["RMSE"]:.12e}')
    print(f'Plasma fit RSS    = {plasma_rss:.12e}')
    print(f'Plasma fit Omega_A = {a_plasma:.12e} rad/s')
    print(f'Plasma fit sqrt(S_W) = {s_plasma:.12e} rad/s')
    print(f'Converged runs    = {sum(r["success"] for r in records)}/{len(records)}')
    print(f'Fits with RSS within 1e-9 of best = {sum(abs(r["RSS"]-best[0]) < 1e-9 for r in records)}')
    print(f'Fit times {time.time()-start_time:.2f} seconds')


if __name__ == '__main__':
    main()
