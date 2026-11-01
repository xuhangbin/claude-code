"""Design geometry for target absorption curves.

Each curve is passed through the inverse network, then refined on the
transfer-matrix model. The figure shows both spectra on the same axes.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from origami_mpp_tandem.invert import network_proposal, refine_geometry
from origami_mpp_tandem.nets import InverseNet
from origami_mpp_tandem.physics import absorption, frequency_axis, sample_params, stinson_fluid
from origami_mpp_tandem.train import DEFAULT_OUT


def load_inverse(path: Path) -> tuple[InverseNet, np.ndarray]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    freq = np.asarray(blob["freq"], dtype=np.float64)
    inverse = InverseNet(freq.size)
    inverse.load_state_dict(blob["inverse"])
    inverse.eval()
    return inverse, freq


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert target absorption spectra.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=Path("origami_mpp_tandem/demo_inverse.png"))
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()

    inverse, freq = load_inverse(args.checkpoint)
    if not np.allclose(freq, frequency_axis()):
        raise RuntimeError("checkpoint frequencies differ from the physics model")
    rng = np.random.default_rng(1)
    true_theta = sample_params(args.cases, rng)
    fluid = stinson_fluid(freq)
    targets = absorption(true_theta, freq, fluid=fluid)

    proposed = []
    refined = []
    for i in range(args.cases):
        guess = network_proposal(inverse, targets[i])
        proposed.append(absorption(guess, freq, fluid=fluid))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = refine_geometry(targets[i], freq, proposal=guess, seed=i, fluid=fluid)
        refined.append(absorption(found, freq, fluid=fluid))
        print(
            f"case {i + 1}  network MAE {np.mean(np.abs(proposed[-1] - targets[i])):.4f}"
            f"  refined MAE {np.mean(np.abs(refined[-1] - targets[i])):.4f}"
        )

    fig, axes = plt.subplots(args.cases, 1, figsize=(8, 2.6 * args.cases), sharex=True)
    if args.cases == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(freq, targets[i], color="0.15", lw=1.8, label="target")
        ax.plot(freq, proposed[i], color="C3", lw=1.3, ls=":", label="inverse network")
        ax.plot(freq, refined[i], color="C0", lw=1.6, ls="--", label="network + transfer matrix")
        ax.set_ylim(0, 1)
        ax.set_ylabel("absorption")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("frequency / Hz")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
