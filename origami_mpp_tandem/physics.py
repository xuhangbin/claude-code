"""Vectorized transfer-matrix model of a four-cell origami micro-perforated absorber.

This is the forward map in the tandem network: design parameters to the
forward-incidence absorption spectrum. Geometry and formulas follow the
user's MATLAB script (Maa micro-perforated impedance, origami cavity volume,
Stinson effective fluid, series transfer matrices, parallel admittance).
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

RHO0 = 1.29
C0 = 343.0
ETA = 1.89e-5
KAPPA = 0.026
CV = 712.0
GAMMA = 1.4
P0 = RHO0 * C0**2 / GAMMA
Z0 = RHO0 * C0
NU = ETA / RHO0
MU_BAR = KAPPA / (RHO0 * CV)

W = 30e-3
L = 30e-3
S = W * L
NC = 4
N_PANELS = 3
N_CAVITIES = 2
T_PANEL = 2e-3

A_MM = 30.0
HC_MM = 20.0
N0 = 4
BETA_DEG = 45.0
H_MAX_MM = N0 * HC_MM  # 80 mm

# Design box used for sampling. Diameters match the scale of the MATLAB example.
D_MIN = 1.0e-3
D_MAX = 8.0e-3
H_MIN = 5.0e-3
H_MAX = 80.0e-3

N_PARAMS = NC * N_PANELS + NC * N_CAVITIES  # 12 diameters + 8 cavity heights


def frequency_axis(f_min: float = 100.0, f_max: float = 700.0, df: float = 1.0) -> np.ndarray:
    return np.arange(f_min, f_max + 0.5 * df, df, dtype=np.float64)


def pack_params(d: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Pack diameters (..., 4, 3) and heights (..., 4, 2) into (..., 20)."""
    d = np.asarray(d, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return np.concatenate([d.reshape(d.shape[:-2] + (NC * N_PANELS,)), h.reshape(h.shape[:-2] + (NC * N_CAVITIES,))], axis=-1)


def unpack_params(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    flat = theta.reshape(-1, N_PARAMS)
    d = flat[:, : NC * N_PANELS].reshape(-1, NC, N_PANELS)
    h = flat[:, NC * N_PANELS :].reshape(-1, NC, N_CAVITIES)
    if theta.ndim == 1:
        return d[0], h[0]
    return d, h


def normalize_params(theta: np.ndarray) -> np.ndarray:
    low, high = _bounds()
    return (np.asarray(theta, dtype=np.float64) - low) / (high - low)


def denormalize_params(unit: np.ndarray) -> np.ndarray:
    low, high = _bounds()
    return low + np.asarray(unit, dtype=np.float64) * (high - low)


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.concatenate([np.full(NC * N_PANELS, D_MIN), np.full(NC * N_CAVITIES, H_MIN)])
    high = np.concatenate([np.full(NC * N_PANELS, D_MAX), np.full(NC * N_CAVITIES, H_MAX)])
    return low, high


def sample_params(n: int, rng: np.random.Generator) -> np.ndarray:
    low, high = _bounds()
    return rng.uniform(low, high, size=(n, N_PARAMS))


def equivalent_height(h_m: np.ndarray) -> np.ndarray:
    """Origami cavity height (m) to equivalent acoustic height (m)."""
    h_mm = np.asarray(h_m, dtype=np.float64) * 1e3
    if np.any(h_mm <= 0) or np.any(h_mm > H_MAX_MM + 1e-9):
        raise ValueError(f"cavity height must satisfy 0 < H <= {H_MAX_MM:.2f} mm")
    h = h_mm / N0
    q = np.sqrt(np.maximum(0.0, HC_MM**2 - h**2))
    v11 = A_MM * h * q / 16.0
    v12 = h * (A_MM - HC_MM) * q / (16.0 * np.tan(np.deg2rad(BETA_DEG)))
    v13 = h * q / 24.0 * (HC_MM / np.tan(np.deg2rad(BETA_DEG)) - q / 2.0)
    v_mm3 = 8.0 * N0 * (A_MM**2 * h / 8.0 - v11 - v12 - v13)
    if np.any(v_mm3 <= 0):
        raise ValueError("origami cavity volume is non-positive")
    return v_mm3 * 1e-9 / S


def stinson_fluid(freq: np.ndarray, n_terms: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Effective wavenumber and characteristic impedance, shape (Nf,)."""
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    m = np.arange(n_terms + 1, dtype=np.float64)
    n = np.arange(n_terms + 1, dtype=np.float64)
    am = (2.0 * m + 1.0) * np.pi / W
    bn = (2.0 * n + 1.0) * np.pi / L
    am2 = am**2
    bn2 = bn**2
    lambda2 = am2[:, None] + bn2[None, :]
    coef = lambda2 / (am2[:, None] * bn2[None, :])

    nf = omega.size
    sv = np.zeros(nf, dtype=np.complex128)
    st = np.zeros(nf, dtype=np.complex128)
    # Chunk modes so the (mode, frequency) temporary stays modest.
    step = 40
    for i0 in range(0, n_terms + 1, step):
        i1 = min(n_terms + 1, i0 + step)
        lam = lambda2[i0:i1]
        c = coef[i0:i1]
        sv += np.sum(c[:, :, None] / (lam[:, :, None] + 1j * omega[None, None, :] / NU), axis=(0, 1))
        st += np.sum(c[:, :, None] / (lam[:, :, None] + 1j * omega[None, None, :] / MU_BAR), axis=(0, 1))

    fv = 64.0 * sv / (W**2 * L**2)
    ft = 64.0 * st / (W**2 * L**2)
    rho_eff = RHO0 / (1.0 - fv)
    k_eff_mod = GAMMA * P0 / (1.0 + (GAMMA - 1.0) * ft)
    k_eff = omega * np.sqrt(rho_eff / k_eff_mod)
    z_eff = np.sqrt(rho_eff * k_eff_mod)
    return k_eff, z_eff


def _mpp_impedance(d: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Maa impedance. d: (B, 4, 3), omega: (Nf,) -> (B, 4, 3, Nf)."""
    y = (d[..., None] / 2.0) / np.sqrt(ETA / (RHO0 * omega))
    q = y * np.sqrt(-1j)
    # jv is fine for these moderate complex arguments.
    f_visc = 1.0 - 2.0 * jv(1, q) / (q * jv(0, q))
    phi = np.pi * d**2 / (4.0 * S)
    z = (
        1j * omega * RHO0 * T_PANEL / (phi[..., None] * f_visc)
        + np.sqrt(2.0) * ETA * y / (phi[..., None] * d[..., None])
        + 1j * 0.85 * omega * RHO0 * d[..., None] / phi[..., None]
    )
    return z


def _panel(z: np.ndarray) -> np.ndarray:
    eye = np.zeros(z.shape + (2, 2), dtype=np.complex128)
    eye[..., 0, 0] = 1.0
    eye[..., 1, 1] = 1.0
    eye[..., 0, 1] = z
    return eye


def _cavity(k: np.ndarray, zc: np.ndarray, height: np.ndarray) -> np.ndarray:
    kh = k * height[..., None]
    c = np.cos(kh)
    s = np.sin(kh)
    mat = np.empty(c.shape + (2, 2), dtype=np.complex128)
    mat[..., 0, 0] = c
    mat[..., 1, 1] = c
    mat[..., 0, 1] = 1j * zc * s
    mat[..., 1, 0] = 1j * s / zc
    return mat


def absorption(
    theta: np.ndarray,
    freq: np.ndarray | None = None,
    fluid: tuple[np.ndarray, np.ndarray] | None = None,
    n_terms: int = 200,
) -> np.ndarray:
    """Forward-incidence absorption. theta (20,) or (B, 20) -> (Nf,) or (B, Nf)."""
    if freq is None:
        freq = frequency_axis()
    freq = np.asarray(freq, dtype=np.float64)
    single = np.asarray(theta).ndim == 1
    d, h = unpack_params(np.atleast_2d(theta))
    if fluid is None:
        k_eff, z_eff = stinson_fluid(freq, n_terms=n_terms)
    else:
        k_eff, z_eff = fluid

    omega = 2.0 * np.pi * freq
    zp = _mpp_impedance(d, omega)
    h_eq = equivalent_height(h)
    # Move frequency next to the 2x2 so the five matrices multiply directly.
    t = (
        _panel(zp[:, :, 0, :])
        @ _cavity(k_eff, z_eff, h_eq[:, :, 0])
        @ _panel(zp[:, :, 1, :])
        @ _cavity(k_eff, z_eff, h_eq[:, :, 1])
        @ _panel(zp[:, :, 2, :])
    )
    a = t[..., 0, 0]
    b = t[..., 0, 1]
    c = t[..., 1, 0]
    dd = t[..., 1, 1]
    ys = np.empty_like(t)
    ys[..., 0, 0] = dd / b
    ys[..., 0, 1] = c - a * dd / b
    ys[..., 1, 0] = 1.0 / b
    ys[..., 1, 1] = -a / b
    y = ys.mean(axis=1)

    y11, y12, y21, y22 = y[..., 0, 0], y[..., 0, 1], y[..., 1, 0], y[..., 1, 1]
    aa = -y22 / y21
    bb = 1.0 / y21
    cc = y12 - y11 * y22 / y21
    d4 = y11 / y21
    den = aa + bb / Z0 + cc * Z0 + d4
    rf = (aa + bb / Z0 - cc * Z0 - d4) / den
    tau = 2.0 / den
    alpha = 1.0 - np.abs(rf) ** 2 - np.abs(tau) ** 2
    alpha = np.real(alpha)
    return alpha[0] if single else alpha


def example_params() -> np.ndarray:
    """The geometry written into the user's MATLAB script."""
    d_mm = np.array(
        [
            [7.652, 6.683, 6.789],
            [3.979, 1.111, 2.342],
            [7.721, 5.829, 6.170],
            [6.724, 7.231, 7.377],
        ],
        dtype=np.float64,
    )
    h_mm = np.array(
        [
            [77.732, 80.0],
            [76.105, 23.526],
            [78.047, 79.630],
            [78.042, 79.969],
        ],
        dtype=np.float64,
    )
    return pack_params(d_mm * 1e-3, h_mm * 1e-3)
