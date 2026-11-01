"""Train the tandem pair and design a geometry for a target absorption spectrum.

The forward network is trained on spectra from the transfer-matrix model.
It is then frozen. The inverse network is trained only through that frozen
network, so a target spectrum is mapped to geometry whose predicted spectrum
matches the target. One spectrum can come from many geometries; matching the
spectrum avoids forcing a single arbitrary parameter vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from origami_mpp_tandem.nets import ForwardNet, InverseNet
from origami_mpp_tandem.physics import (
    absorption,
    denormalize_params,
    frequency_axis,
    normalize_params,
    sample_params,
    stinson_fluid,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "checkpoints" / "tandem.pt"


def build_dataset(n: int, seed: int, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = sample_params(n, rng)
    fluid = stinson_fluid(freq)
    spectra = np.empty((n, freq.size), dtype=np.float32)
    batch = 256
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        spectra[i0:i1] = absorption(theta[i0:i1], freq, fluid=fluid).astype(np.float32)
    unit = normalize_params(theta).astype(np.float32)
    return unit, spectra


def _loader(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool):
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt),
        batch_size=batch,
        shuffle=shuffle,
    )


def train_forward(model: ForwardNet, train, val, epochs: int, lr: float, device: torch.device) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
        schedule.step()
        mae = evaluate_forward(model, val, device)
        if mae < best:
            best = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"forward epoch {epoch:3d}  mse {total / count:.6f}  val MAE {mae:.5f}")
    model.load_state_dict(best_state)


@torch.no_grad()
def evaluate_forward(model: ForwardNet, loader, device: torch.device) -> float:
    model.eval()
    err = 0.0
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        err += torch.abs(pred - yb).sum().item()
        count += yb.numel()
    return err / count


def train_inverse(inverse: InverseNet, forward: ForwardNet, train, val, epochs: int, lr: float, device: torch.device) -> None:
    forward.eval()
    for parameter in forward.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.Adam(inverse.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        inverse.train()
        total = 0.0
        count = 0
        for _xb, yb in train:
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = forward(inverse(yb))
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * yb.shape[0]
            count += yb.shape[0]
        schedule.step()
        mae = evaluate_tandem(inverse, forward, val, device)
        if mae < best:
            best = mae
            best_state = {k: v.detach().cpu().clone() for k, v in inverse.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"inverse epoch {epoch:3d}  tandem mse {total / count:.6f}  val MAE {mae:.5f}")
    inverse.load_state_dict(best_state)


@torch.no_grad()
def evaluate_tandem(inverse: InverseNet, forward: ForwardNet, loader, device: torch.device) -> float:
    inverse.eval()
    forward.eval()
    err = 0.0
    count = 0
    for _xb, yb in loader:
        yb = yb.to(device)
        pred = forward(inverse(yb))
        err += torch.abs(pred - yb).sum().item()
        count += yb.numel()
    return err / count


def physics_mae(inverse: InverseNet, spectra: np.ndarray, freq: np.ndarray, device: torch.device) -> float:
    inverse.eval()
    with torch.no_grad():
        unit = inverse(torch.from_numpy(spectra).to(device)).cpu().numpy()
    theta = denormalize_params(unit)
    fluid = stinson_fluid(freq)
    pred = absorption(theta, freq, fluid=fluid)
    return float(np.mean(np.abs(pred - spectra)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the origami tandem inverse-design networks.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--forward-epochs", type=int, default=200)
    parser.add_argument("--inverse-epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = frequency_axis()
    print(f"device {device}  frequencies {freq.size}  samples {args.samples}")
    unit, spectra = build_dataset(args.samples, args.seed, freq)
    n_val = max(1, args.samples // 10)
    n_test = n_val
    n_train = args.samples - n_val - n_test
    splits = {
        "train": (unit[:n_train], spectra[:n_train]),
        "val": (unit[n_train : n_train + n_val], spectra[n_train : n_train + n_val]),
        "test": (unit[n_train + n_val :], spectra[n_train + n_val :]),
    }
    train_loader = _loader(*splits["train"], args.batch, True)
    val_loader = _loader(*splits["val"], args.batch, False)

    forward = ForwardNet(freq.size).to(device)
    inverse = InverseNet(freq.size).to(device)
    train_forward(forward, train_loader, val_loader, args.forward_epochs, args.lr, device)
    train_inverse(inverse, forward, train_loader, val_loader, args.inverse_epochs, args.lr, device)

    test_spectra = splits["test"][1]
    net_mae = physics_mae(inverse, test_spectra, freq, device)
    print(f"test absorption MAE after transfer-matrix re-evaluation: {net_mae:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "forward": forward.cpu().state_dict(),
            "inverse": inverse.cpu().state_dict(),
            "freq": freq,
            "test_mae": net_mae,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
