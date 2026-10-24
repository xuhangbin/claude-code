"""用传递矩阵模型生成样本，训练频谱代理网络。

示例：

    python -m acoustic_nn.train --samples 4000 --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from acoustic_nn.model import SpectralSurrogate
from acoustic_nn.physics import (
    Air,
    duct_wavenumber,
    feature_bounds,
    frequency_axis,
    normalize_features,
    pack_geometry,
    sample_geometries,
    spectra_targets,
    transfer_response,
)

CHANNEL_NAMES = ("alpha_f", "alpha_b", "ep_diff")


def generate_dataset(
    count: int,
    rng: np.random.Generator,
    frequency: np.ndarray,
    wavenumber: np.ndarray,
    impedance: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回归一化几何 ``(count, 32)`` 和频谱 ``(count, 3, frequency)``。"""

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lower, upper = feature_bounds()
    filled = 0
    while filled < count:
        batch = min(chunk, count - filled)
        # 多抽一些，丢掉传递矩阵里出现非有限值的样本。
        draw = batch + max(8, batch // 5)
        diameter, thickness, depth = sample_geometries(draw, rng)
        response = transfer_response(
            diameter,
            thickness,
            depth,
            frequency,
            wavenumber=wavenumber,
            characteristic_impedance=impedance,
        )
        spectrum = spectra_targets(response)
        finite = np.isfinite(spectrum).all(axis=(1, 2))
        packed = normalize_features(pack_geometry(diameter, thickness, depth), lower, upper)
        packed = packed[finite][:batch]
        spectrum = spectrum[finite][:batch]
        if len(spectrum) == 0:
            raise RuntimeError("这一批样本全部非有限，请检查几何边界")
        features.append(packed)
        targets.append(spectrum)
        filled += len(spectrum)
    return np.concatenate(features, axis=0)[:count], np.concatenate(targets, axis=0)[:count]


def channel_mae(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = (prediction - target).abs().mean(dim=(0, 2))
    return {name: float(error[index]) for index, name in enumerate(CHANNEL_NAMES)}


def train(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    frequency = frequency_axis()
    print(f"预计算 Stinson 等效参数，频率点数 {frequency.size}，模态 N={args.stinson_terms}")
    wavenumber, impedance = duct_wavenumber(frequency, Air(), stinson_terms=args.stinson_terms)

    print(f"生成训练集 {args.samples} 条，验证集 {args.val_samples} 条")
    train_x, train_y = generate_dataset(
        args.samples, rng, frequency, wavenumber, impedance, args.chunk
    )
    val_x, val_y = generate_dataset(
        args.val_samples, rng, frequency, wavenumber, impedance, args.chunk
    )
    device = torch.device("cpu")
    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)

    model = SpectralSurrogate(n_freq=frequency.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_x_t.shape[0])
        total = 0.0
        seen = 0
        for start in range(0, order.numel(), args.batch_size):
            index = order[start : start + args.batch_size]
            prediction = model(train_x_t[index])
            loss = loss_fn(prediction, train_y_t[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * index.numel()
            seen += index.numel()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x_t)
            val_loss = float(loss_fn(val_prediction, val_y_t))
            mae = channel_mae(val_prediction, val_y_t)
        print(
            f"epoch {epoch:03d}  train_mse={total / seen:.6f}  val_mse={val_loss:.6f}  "
            f"MAE αf={mae['alpha_f']:.4f}  αb={mae['alpha_b']:.4f}  EP={mae['ep_diff']:.4f}"
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"验证集连续 {args.patience} 轮没有下降，停止训练")
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可用参数")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_prediction = model(val_x_t)
        final_mae = channel_mae(final_prediction, val_y_t)

    lower, upper = feature_bounds()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "val_mse": best_val,
            "val_mae": final_mae,
            "frequency": frequency,
            "bounds_min": lower,
            "bounds_max": upper,
            "channels": CHANNEL_NAMES,
            "samples": args.samples,
            "seed": args.seed,
            "stinson_terms": args.stinson_terms,
        },
        destination,
    )
    print(
        f"已保存 {destination}  最优 val_mse={best_val:.6f}  "
        f"MAE αf={final_mae['alpha_f']:.4f}  αb={final_mae['alpha_b']:.4f}  "
        f"EP={final_mae['ep_diff']:.4f}"
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练微穿孔板频谱代理网络")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stinson-terms", type=int, default=200)
    parser.add_argument(
        "--output",
        default="acoustic_nn/checkpoints/surrogate.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
