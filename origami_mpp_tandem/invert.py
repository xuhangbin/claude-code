"""Find geometry for a target absorption spectrum.

The tandem network is the same encoder-decoder layout as the figure: the
inverse network proposes normalized parameters, and the frozen forward network
is what it was trained to match. Twenty geometric parameters sampled over the
full box are too sparse for that network to place the resonance peaks exactly,
so the proposal is then refined on the transfer-matrix model itself.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.stats.qmc import LatinHypercube

from origami_mpp_tandem.physics import (
    N_PARAMS,
    absorption,
    denormalize_params,
    frequency_axis,
    normalize_params,
    stinson_fluid,
)


def _objective(unit: np.ndarray, target: np.ndarray, freq: np.ndarray, fluid) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float64)
    if unit.ndim == 1:
        spectrum = absorption(denormalize_params(unit), freq, fluid=fluid)
        return np.mean((spectrum - target) ** 2)
    spectra = absorption(denormalize_params(unit.T), freq, fluid=fluid)
    return np.mean((spectra - target) ** 2, axis=1)


def refine_geometry(
    target: np.ndarray,
    freq: np.ndarray | None = None,
    proposal: np.ndarray | None = None,
    seed: int = 0,
    popsize: int = 60,
    maxiter: int = 25,
    fluid=None,
) -> np.ndarray:
    """Return physical parameters (20,) whose absorption matches ``target``."""
    if freq is None:
        freq = frequency_axis()
    target = np.asarray(target, dtype=np.float64)
    if target.shape != freq.shape:
        raise ValueError(f"target length {target.shape} does not match {freq.size} frequencies")
    if fluid is None:
        fluid = stinson_fluid(freq)

    population = LatinHypercube(d=N_PARAMS, seed=seed).random(popsize)
    if proposal is not None:
        population[0] = np.clip(normalize_params(np.asarray(proposal, dtype=np.float64)), 0.0, 1.0)

    result = differential_evolution(
        _objective,
        bounds=[(0.0, 1.0)] * N_PARAMS,
        args=(target, freq, fluid),
        init=population,
        seed=seed,
        maxiter=maxiter,
        mutation=0.5,
        recombination=0.7,
        polish=False,
        vectorized=True,
        atol=0.0,
        tol=1e-8,
        workers=1,
    )
    return denormalize_params(result.x)


def network_proposal(inverse: torch.nn.Module, target: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")
    inverse.eval()
    with torch.no_grad():
        unit = inverse(torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device))
    return denormalize_params(unit.cpu().numpy())
