"""rx0 bathymetry smoothing via linear programming.

Adapted from pygetm.domain.Domain.smooth() and Domain.get_rx0() in
GETM/pygetm/python/pygetm/domain.py (lines 1139–1264), with the GETM
staggered-grid indexing removed so the functions operate directly on a
standard 2-D depth array.

The slope factor rx0 at an interface between adjacent wet cells with depths
H1 and H2 is defined as (Haney 1991):

    rx0 = |H1 - H2| / (H1 + H2)

Smoothing finds the minimum-L1-norm depth corrections that bring every
interface below the target rx0, using scipy.optimize.linprog.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_rx0(
    depth: npt.NDArray[np.float64],
    mask: npt.NDArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the rx0 slope factor at every U (x) and V (y) interface.

    Parameters
    ----------
    depth : ndarray [ny, nx]
        Positive-down depth in metres.
    mask : ndarray [ny, nx]
        Ocean mask (1 = ocean, 0 = land).

    Returns
    -------
    rx0_u : ndarray [ny, nx-1]
        Slope factor at x-direction (U) interfaces.
    rx0_v : ndarray [ny-1, nx]
        Slope factor at y-direction (V) interfaces.
    """
    wet = mask.astype(bool)
    H = depth.copy()
    H[~wet] = 0.0

    H1_u = H[:, :-1]
    H2_u = H[:, 1:]
    denom_u = H1_u + H2_u
    # np.where evaluates both branches before selecting, triggering a divide-by-zero
    # warning even for masked cells.  np.divide with where= avoids this.
    rx0_u = np.divide(np.abs(H1_u - H2_u), denom_u,
                      out=np.zeros_like(denom_u), where=denom_u > 0)
    rx0_u[~wet[:, :-1] | ~wet[:, 1:]] = 0.0

    H1_v = H[:-1, :]
    H2_v = H[1:, :]
    denom_v = H1_v + H2_v
    rx0_v = np.divide(np.abs(H1_v - H2_v), denom_v,
                      out=np.zeros_like(denom_v), where=denom_v > 0)
    rx0_v[~wet[:-1, :] | ~wet[1:, :]] = 0.0

    return rx0_u.astype(np.float64), rx0_v.astype(np.float64)


def max_rx0(
    depth: npt.NDArray[np.float64],
    mask: npt.NDArray,
) -> float:
    """Maximum rx0 over all wet interfaces."""
    rx0_u, rx0_v = compute_rx0(depth, mask)
    return float(max(rx0_u.max(initial=0.0), rx0_v.max(initial=0.0)))


def smooth_rx0(
    depth: npt.NDArray[np.float64],
    mask: npt.NDArray,
    rx0: float = 0.2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Smooth bathymetry by reducing the slope factor to *rx0* everywhere.

    Finds the minimum-L1-norm correction to *depth* such that the rx0
    criterion is satisfied at every wet U and V interface.  Uses
    scipy.optimize.linprog with sparse constraints.

    Parameters
    ----------
    depth : ndarray [ny, nx]
        Input depth (positive-down, m).
    mask : ndarray [ny, nx]
        Ocean mask (1 = ocean, 0 = land).
    rx0 : float
        Target maximum slope factor.

    Returns
    -------
    depth_smoothed : ndarray [ny, nx]
        Corrected depth field.
    corrections : ndarray [ny, nx]
        Per-cell depth corrections (zero on land).
    """
    import scipy.optimize
    import scipy.sparse

    current = max_rx0(depth, mask)
    if current <= rx0:
        return depth.copy(), np.zeros_like(depth)

    wet = mask.astype(bool)
    nwet = int(wet.sum())
    H = depth[wet]

    # Index map: wet cell → LP variable index; -1 = land
    iwet = np.full(wet.shape, -1, dtype=np.intp)
    iwet[wet] = np.arange(nwet)

    # Wet interfaces
    uwet = wet[:, 1:] & wet[:, :-1]  # [ny, nx-1]
    vwet = wet[1:, :] & wet[:-1, :]  # [ny-1, nx]
    nu = int(uwet.sum())
    nv = int(vwet.sum())
    n = nu + nv
    nconstraints = n * 2 + nwet * 2

    A_values = np.empty((nconstraints, 2))
    A_i = np.empty_like(A_values, dtype=np.intp)
    A_j = np.empty_like(A_values, dtype=np.intp)
    b = np.zeros(nconstraints)

    # Slope constraints: (1-rx0)*H1' + (-1-rx0)*H2' < -(1-rx0)*H1 - (-1-rx0)*H2
    #                    (-1-rx0)*H1' + (1-rx0)*H2' < -(-1-rx0)*H1 - (1-rx0)*H2
    A_values[:n, 0] = 1.0 - rx0
    A_values[:n, 1] = -1.0 - rx0
    A_values[n : 2 * n, 0] = -1.0 - rx0
    A_values[n : 2 * n, 1] = 1.0 - rx0
    A_i[:, :] = np.arange(nconstraints)[:, np.newaxis]

    # U interfaces (x-direction)
    A_j[:nu, 0] = iwet[:, :-1][uwet]
    A_j[:nu, 1] = iwet[:, 1:][uwet]
    # V interfaces (y-direction)
    A_j[nu:n, 0] = iwet[:-1, :][vwet]
    A_j[nu:n, 1] = iwet[1:, :][vwet]
    # Mirrored block for the second inequality set
    A_j[n : 2 * n, :] = A_j[:n, :]

    b[: 2 * n] = -(A_values[: 2 * n] * H[A_j[: 2 * n]]).sum(axis=1)

    # Minimise sum of absolute corrections via auxiliary variable M_i >= |H'_i|
    # LP variables: [H'_0, …, H'_{nwet-1}, M_0, …, M_{nwet-1}]
    A_j[2 * n : 2 * n + nwet, 0] = np.arange(nwet)
    A_j[2 * n : 2 * n + nwet, 1] = np.arange(nwet, 2 * nwet)
    A_j[2 * n + nwet :, :] = A_j[2 * n : 2 * n + nwet, :]
    A_values[2 * n : 2 * n + nwet, 0] = 1.0   # H'_i - M_i <= 0
    A_values[2 * n + nwet :, 0] = -1.0          # -H'_i - M_i <= 0
    A_values[2 * n :, 1] = -1.0

    c = np.zeros(2 * nwet)
    c[nwet:] = 1.0  # minimise sum(M)

    A = scipy.sparse.coo_matrix(
        (A_values.ravel(), (A_i.ravel(), A_j.ravel())),
        shape=(nconstraints, 2 * nwet),
    )
    res = scipy.optimize.linprog(c=c, A_ub=A, b_ub=b, bounds=(None, None))
    if not res.success:
        raise RuntimeError(f"rx0 smoothing LP failed: {res.message}")

    corrections = np.zeros_like(depth)
    corrections[wet] = res.x[:nwet]
    depth_smoothed = depth + corrections

    return depth_smoothed, corrections


def smooth_summary(
    depth_before: npt.NDArray[np.float64],
    depth_after: npt.NDArray[np.float64],
    mask: npt.NDArray,
    corrections: npt.NDArray[np.float64],
    rx0_target: float,
) -> dict:
    """Return a summary dict for the smoothing step."""
    rx0_u_before, rx0_v_before = compute_rx0(depth_before, mask)
    rx0_u_after, rx0_v_after = compute_rx0(depth_after, mask)
    wet = mask.astype(bool)
    modified = np.abs(corrections[wet]) > 1e-4
    return {
        "rx0 target": rx0_target,
        "max rx0 before": f"{max(rx0_u_before.max(initial=0.0), rx0_v_before.max(initial=0.0)):.4f}",
        "max rx0 after": f"{max(rx0_u_after.max(initial=0.0), rx0_v_after.max(initial=0.0)):.4f}",
        "cells modified": int(modified.sum()),
        "max correction (m)": f"{float(np.abs(corrections).max()):.2f}",
        "mean |correction| (m)": f"{float(np.abs(corrections[wet]).mean()):.3f}",
    }
