"""比较代理网络和传递矩阵模型，并沿网络梯度搜索更小的特征值间隔。

示例：

    python -m acoustic_nn.demo --checkpoint acoustic_nn/checkpoints/surrogate.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from acoustic_nn.model import SpectralSurrogate
from acoustic_nn.physics import (
    denormalize_features,
    duct_wavenumber,
    frequency_axis,
    nominal_geometry,
    normalize_features,
    pack_geometry,
    transfer_response,
    unpack_geometry,
)


def load_model(checkpoint_path: Path) -> tuple[SpectralSurrogate, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SpectralSurrogate(n_freq=len(checkpoint["frequency"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_geometry(
    model: SpectralSurrogate,
    diameter: np.ndarray,
    thickness: np.ndarray,
    depth: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    features = normalize_features(
        pack_geometry(diameter, thickness, depth), bounds_min, bounds_max
    )
    with torch.no_grad():
        prediction = model(torch.tensor(features[None, :], dtype=torch.float32))
    return prediction[0].numpy()


def search_lower_gap(
    model: SpectralSurrogate,
    start_normalized: np.ndarray,
    steps: int,
    lr: float,
) -> np.ndarray:
    """从名义几何出发，沿网络梯度缩小特征值间隔。

    约束写在网络自己的预测上：前向吸声平均值不能低于出发点。
    这样搜索不会靠牺牲吸声来换间隔。
    """

    clipped = np.clip(start_normalized, 1e-4, 1.0 - 1e-4)
    logit = np.log(clipped / (1.0 - clipped))
    raw = torch.nn.Parameter(torch.tensor(logit[None, :], dtype=torch.float32))
    anchor = torch.tensor(start_normalized, dtype=torch.float32)
    with torch.no_grad():
        absorption_floor = model(anchor[None])[0, 0].mean()
    optimizer = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        normalized = torch.sigmoid(raw)
        prediction = model(normalized)[0]
        gap = prediction[2].mean()
        absorption = prediction[0].mean()
        loss = gap + torch.relu(absorption_floor - absorption)
        loss = loss + 0.01 * (normalized - anchor).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.sigmoid(raw).detach().numpy()[0]


def plot_comparison(
    frequency: np.ndarray,
    physics: dict[str, np.ndarray],
    prediction: np.ndarray,
    optimized_physics: dict[str, np.ndarray],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    axes[0].plot(frequency, physics["alpha_f"], color="C0", lw=1.8, label="Physics absorption")
    axes[0].plot(frequency, physics["Rf"], color="C3", ls="--", lw=1.4, label="Physics reflection")
    axes[0].plot(frequency, physics["Tr"], color="k", ls=":", lw=1.4, label="Physics transmission")
    axes[0].plot(frequency, prediction[0], color="C0", ls="-.", lw=1.6, label="Network absorption")
    axes[0].plot(
        frequency,
        optimized_physics["alpha_f"],
        color="C1",
        lw=1.4,
        label="Optimized physics absorption",
    )
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("(a) Forward incidence")
    axes[0].legend(loc="best")
    axes[0].grid(True)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(frequency, physics["alpha_b"], color="C0", lw=1.8, label="Physics absorption")
    axes[1].plot(frequency, physics["Rb"], color="C3", ls="--", lw=1.4, label="Physics reflection")
    axes[1].plot(frequency, physics["Tr"], color="k", ls=":", lw=1.4, label="Physics transmission")
    axes[1].plot(frequency, prediction[1], color="C0", ls="-.", lw=1.6, label="Network absorption")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_title("(b) Backward incidence")
    axes[1].legend(loc="best")
    axes[1].grid(True)
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(frequency, physics["ep_diff"], color="C2", lw=1.8, label="Nominal physics")
    axes[2].plot(frequency, prediction[2], color="C2", ls="-.", lw=1.6, label="Nominal network")
    axes[2].plot(
        frequency,
        optimized_physics["ep_diff"],
        color="C1",
        lw=1.8,
        label="Optimized physics",
    )
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("|lambda+ - lambda-|")
    axes[2].set_title("(c) Rearranged scattering-matrix eigenvalue gap")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def run_demo(args: argparse.Namespace) -> None:
    model, checkpoint = load_model(Path(args.checkpoint))
    frequency = np.asarray(checkpoint["frequency"], dtype=np.float64)
    if not np.allclose(frequency, frequency_axis()):
        raise RuntimeError("检查点里的频率轴与物理模型不一致")
    bounds_min = np.asarray(checkpoint["bounds_min"], dtype=np.float64)
    bounds_max = np.asarray(checkpoint["bounds_max"], dtype=np.float64)

    print("计算名义几何的传递矩阵频谱")
    wavenumber, impedance = duct_wavenumber(
        frequency, stinson_terms=int(checkpoint["stinson_terms"])
    )
    diameter, thickness, depth = nominal_geometry()
    physics = transfer_response(
        diameter,
        thickness,
        depth,
        frequency,
        wavenumber=wavenumber,
        characteristic_impedance=impedance,
    )
    prediction = predict_geometry(
        model, diameter, thickness, depth, bounds_min, bounds_max
    )
    mae_forward = float(np.mean(np.abs(prediction[0] - physics["alpha_f"])))
    mae_backward = float(np.mean(np.abs(prediction[1] - physics["alpha_b"])))
    mae_gap = float(np.mean(np.abs(prediction[2] - physics["ep_diff"])))
    print(
        f"名义几何网络误差  MAE αf={mae_forward:.4f}  αb={mae_backward:.4f}  "
        f"EP={mae_gap:.4f}"
    )
    print(
        f"名义几何物理结果  平均 αf={float(np.mean(physics['alpha_f'])):.4f}  "
        f"平均 αb={float(np.mean(physics['alpha_b'])):.4f}  "
        f"平均 EP={float(np.mean(physics['ep_diff'])):.4f}  "
        f"最小 EP={float(np.min(physics['ep_diff'])):.4f}"
    )

    start = normalize_features(pack_geometry(diameter, thickness, depth), bounds_min, bounds_max)
    optimized = search_lower_gap(model, start, steps=args.steps, lr=args.lr)
    opt_d, opt_t, opt_h = unpack_geometry(
        denormalize_features(optimized, bounds_min, bounds_max)
    )
    optimized_physics = transfer_response(
        opt_d,
        opt_t,
        opt_h,
        frequency,
        wavenumber=wavenumber,
        characteristic_impedance=impedance,
    )
    print(
        f"梯度搜索后的物理结果  平均 αf={float(np.mean(optimized_physics['alpha_f'])):.4f}  "
        f"平均 αb={float(np.mean(optimized_physics['alpha_b'])):.4f}  "
        f"平均 EP={float(np.mean(optimized_physics['ep_diff'])):.4f}  "
        f"最小 EP={float(np.min(optimized_physics['ep_diff'])):.4f}"
    )
    print("搜索后的孔径 mm:")
    print(np.array2string(opt_d * 1e3, precision=3, suppress_small=True))
    print("搜索后的板厚 mm:")
    print(np.array2string(opt_t * 1e3, precision=3, suppress_small=True))
    print("搜索后的腔深 mm:")
    print(np.array2string(opt_h * 1e3, precision=3, suppress_small=True))
    geometry_path = Path(args.geometry_output)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        geometry_path,
        diameter_m=opt_d,
        thickness_m=opt_t,
        depth_m=opt_h,
        nominal_diameter_m=diameter,
        nominal_thickness_m=thickness,
        nominal_depth_m=depth,
    )
    plot_comparison(
        frequency,
        physics,
        prediction,
        optimized_physics,
        Path(args.figure),
    )
    print(f"图已写入 {args.figure}")
    print(f"搜索到的几何已写入 {args.geometry_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检验代理网络并搜索更小的特征值间隔")
    parser.add_argument("--checkpoint", default="acoustic_nn/checkpoints/surrogate.pt")
    parser.add_argument("--figure", default="acoustic_nn/figures/nominal_comparison.png")
    parser.add_argument("--geometry-output", default="acoustic_nn/figures/optimized_geometry.npz")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
