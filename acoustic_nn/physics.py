"""四单胞微穿孔板传递矩阵模型。

对应原 MATLAB 脚本的计算顺序：

1. 空气参数与频率
2. 微穿孔板阻抗（含端部修正）
3. 矩形腔的 Full Stinson 等效密度与体积模量
4. 单胞传递矩阵 ``Tp1 * Tc1 * Tp2 * Tc2 * Tp3``
5. 面积加权导纳并联，再恢复总传递矩阵
6. 前后向反射、透射、吸声，以及重排散射矩阵的特征值间隔

重排散射矩阵为 ``S = [[tau, rb], [rf, tau]]``。它的特征值间隔
``|λ+ - λ-| = 2 |sqrt(rf * rb)|`` 并不等同于标准端口散射矩阵的奇异点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv

# 与原脚本一致的采样范围
F_MIN_HZ = 100.0
F_MAX_HZ = 500.0
F_STEP_HZ = 1.0

# 原脚本里的单胞截面与孔数
WIDTH_M = 30e-3
LENGTH_M = 30e-3
N_CELLS = 4
N_PLATES = 3
N_CAVITIES = 2

# 训练采样时使用的几何边界。名义设计落在这个盒子内部。
D_MIN_M = 0.10e-3
D_MAX_M = 4.00e-3
T_MIN_M = 0.50e-3
T_MAX_M = 3.00e-3
H_MIN_M = 20.0e-3
H_MAX_M = 80.0e-3

# 每个单胞打包成 [d1, d2, d3, t1, t2, t3, H1, H2]
FEATURES_PER_CELL = 8
N_FEATURES = N_CELLS * FEATURES_PER_CELL


@dataclass(frozen=True)
class Air:
    """空气参数，数值与原脚本相同。"""

    rho0: float = 1.29
    c0: float = 343.0
    eta: float = 1.89e-5
    kappa: float = 0.026
    Cv: float = 712.0
    gamma: float = 1.4

    @property
    def pressure(self) -> float:
        return self.rho0 * self.c0**2 / self.gamma

    @property
    def impedance(self) -> float:
        return self.rho0 * self.c0

    @property
    def nu(self) -> float:
        return self.eta / self.rho0

    @property
    def mu_bar(self) -> float:
        return self.kappa / (self.rho0 * self.Cv)


def frequency_axis(
    f_min: float = F_MIN_HZ,
    f_max: float = F_MAX_HZ,
    step: float = F_STEP_HZ,
) -> np.ndarray:
    """与 MATLAB ``f = 100:1:500`` 相同的频率轴，单位 Hz。"""

    count = int(round((f_max - f_min) / step))
    return f_min + step * np.arange(count + 1, dtype=np.float64)


def nominal_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """原脚本中的孔径、板厚和腔深，单位 m。"""

    diameter = 1e-3 * np.array(
        [
            [2.43, 1.6628, 0.243],
            [3.6, 1.5, 0.26],
            [3.15, 0.79, 0.15],
            [2.98, 0.56, 0.15],
        ],
        dtype=np.float64,
    )
    thickness = 1e-3 * 2.0 * np.ones((N_CELLS, N_PLATES), dtype=np.float64)
    depth = 1e-3 * 47.0 * np.ones((N_CELLS, N_CAVITIES), dtype=np.float64)
    return diameter, thickness, depth


def feature_bounds() -> tuple[np.ndarray, np.ndarray]:
    """几何特征的下界和上界，形状 ``(32,)``，单位 m。"""

    lower = np.array(
        [D_MIN_M, D_MIN_M, D_MIN_M, T_MIN_M, T_MIN_M, T_MIN_M, H_MIN_M, H_MIN_M],
        dtype=np.float64,
    )
    upper = np.array(
        [D_MAX_M, D_MAX_M, D_MAX_M, T_MAX_M, T_MAX_M, T_MAX_M, H_MAX_M, H_MAX_M],
        dtype=np.float64,
    )
    return np.tile(lower, N_CELLS), np.tile(upper, N_CELLS)


def pack_geometry(
    diameter: np.ndarray,
    thickness: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """把几何数组打包成网络输入，形状 ``(..., 32)``。"""

    diameter = np.asarray(diameter, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    parts = np.concatenate([diameter, thickness, depth], axis=-1)
    return parts.reshape(parts.shape[:-2] + (N_FEATURES,))


def unpack_geometry(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``pack_geometry`` 的逆变换。"""

    features = np.asarray(features, dtype=np.float64)
    flat = features.reshape(features.shape[:-1] + (N_CELLS, FEATURES_PER_CELL))
    diameter = flat[..., 0:3]
    thickness = flat[..., 3:6]
    depth = flat[..., 6:8]
    return diameter, thickness, depth


def normalize_features(
    features: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """把几何特征线性映射到 ``[0, 1]``。"""

    if lower is None or upper is None:
        lower, upper = feature_bounds()
    return (np.asarray(features, dtype=np.float64) - lower) / (upper - lower)


def denormalize_features(
    normalized: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """``normalize_features`` 的逆变换。"""

    if lower is None or upper is None:
        lower, upper = feature_bounds()
    return lower + np.asarray(normalized, dtype=np.float64) * (upper - lower)


def duct_wavenumber(
    frequency: np.ndarray,
    air: Air = Air(),
    width: float = WIDTH_M,
    length: float = LENGTH_M,
    stinson_terms: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """矩形截面腔的 Full Stinson 等效波数和特性阻抗。

    ``stinson_terms`` 与原脚本的 ``N`` 相同，求和指标从 0 取到 N。
    结果只依赖频率和截面，一组几何样本可以共用。
    """

    omega = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64)
    mode_n = np.arange(stinson_terms + 1, dtype=np.float64)
    beta = (2.0 * mode_n + 1.0) * np.pi / length
    viscous = np.zeros(omega.shape, dtype=np.complex128)
    thermal = np.zeros(omega.shape, dtype=np.complex128)

    for mode_index in range(stinson_terms + 1):
        alpha = (2.0 * mode_index + 1.0) * np.pi / width
        lambda2 = alpha**2 + beta**2
        weight = 1.0 / (alpha**2 * beta**2)
        numerator = weight * lambda2
        viscous += np.sum(
            numerator[:, None] / (lambda2[:, None] + 1j * omega[None, :] / air.nu),
            axis=0,
        )
        thermal += np.sum(
            numerator[:, None]
            / (lambda2[:, None] + 1j * omega[None, :] / air.mu_bar),
            axis=0,
        )

    scale = 64.0 / (width**2 * length**2)
    viscous_factor = scale * viscous
    thermal_factor = scale * thermal
    density = air.rho0 / (1.0 - viscous_factor)
    modulus = air.gamma * air.pressure / (1.0 + (air.gamma - 1.0) * thermal_factor)
    wavenumber = omega * np.sqrt(density / modulus)
    impedance = np.sqrt(density * modulus)
    return wavenumber, impedance


def _plate_matrix(impedance: np.ndarray) -> np.ndarray:
    matrix = np.zeros(impedance.shape + (2, 2), dtype=np.complex128)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = 1.0
    matrix[..., 0, 1] = impedance
    return matrix


def _cavity_matrix(
    wavenumber: np.ndarray,
    impedance: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    phase = wavenumber * depth[..., None]
    cosine = np.cos(phase)
    sine = np.sin(phase)
    matrix = np.zeros(cosine.shape + (2, 2), dtype=np.complex128)
    matrix[..., 0, 0] = cosine
    matrix[..., 1, 1] = cosine
    matrix[..., 0, 1] = 1j * impedance * sine
    matrix[..., 1, 0] = 1j * sine / impedance
    return matrix


def _ensure_batch(
    diameter: np.ndarray,
    thickness: np.ndarray,
    depth: np.ndarray,
    hole_count: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    diameter = np.asarray(diameter, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    single = diameter.ndim == 2
    if single:
        diameter = diameter[None, ...]
        thickness = thickness[None, ...]
        depth = depth[None, ...]
    if diameter.shape[-2:] != (N_CELLS, N_PLATES):
        raise ValueError("孔径数组的最后两维必须是 (4, 3)")
    if thickness.shape != diameter.shape:
        raise ValueError("板厚数组必须与孔径数组同形")
    if depth.shape != diameter.shape[:-1] + (N_CAVITIES,):
        raise ValueError("腔深数组的最后两维必须是 (4, 2)")
    if hole_count is None:
        hole_count = np.ones_like(diameter)
    else:
        hole_count = np.asarray(hole_count, dtype=np.float64)
        if hole_count.ndim == 2:
            hole_count = np.broadcast_to(hole_count, diameter.shape).copy()
        elif single and hole_count.ndim == 3:
            pass
        if hole_count.shape != diameter.shape:
            raise ValueError("孔数数组必须与孔径数组同形")
    return diameter, thickness, depth, hole_count, single


def microperforated_impedance(
    diameter: np.ndarray,
    thickness: np.ndarray,
    porosity: np.ndarray,
    omega: np.ndarray,
    air: Air,
) -> np.ndarray:
    """微穿孔板阻抗，包含黏性管内项和两侧端部修正中的 0.85 项。"""

    shear_length = np.sqrt(air.eta / (air.rho0 * omega))
    shear_number = (diameter[..., None] / 2.0) / shear_length
    argument = shear_number * np.sqrt(-1j)
    bessel_ratio = jv(1, argument) / (argument * jv(0, argument))
    viscosity_factor = 1.0 - 2.0 * bessel_ratio
    return (
        1j * omega * air.rho0 * thickness[..., None] / (porosity[..., None] * viscosity_factor)
        + np.sqrt(2.0) * air.eta * shear_number / (porosity[..., None] * diameter[..., None])
        + 1j * 0.85 * omega * air.rho0 * diameter[..., None] / porosity[..., None]
    )


def transfer_response(
    diameter: np.ndarray,
    thickness: np.ndarray,
    depth: np.ndarray,
    frequency: np.ndarray | None = None,
    air: Air = Air(),
    width: float = WIDTH_M,
    length: float = LENGTH_M,
    hole_count: np.ndarray | None = None,
    stinson_terms: int = 200,
    wavenumber: np.ndarray | None = None,
    characteristic_impedance: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """计算一批几何的吸声、反射、透射和特征值间隔。

    ``diameter`` 的形状是 ``(4, 3)`` 或 ``(batch, 4, 3)``。单样本输入时，
    返回的频谱形状为 ``(frequency,)``；批量输入时为 ``(batch, frequency)``。
    """

    if frequency is None:
        frequency = frequency_axis()
    frequency = np.asarray(frequency, dtype=np.float64)
    omega = 2.0 * np.pi * frequency
    diameter, thickness, depth, hole_count, single = _ensure_batch(
        diameter, thickness, depth, hole_count
    )

    if wavenumber is None or characteristic_impedance is None:
        wavenumber, characteristic_impedance = duct_wavenumber(
            frequency, air, width, length, stinson_terms
        )
    wavenumber = np.asarray(wavenumber)
    characteristic_impedance = np.asarray(characteristic_impedance)

    porosity = np.pi * diameter**2 * hole_count / (4.0 * width * length)
    plate_impedance = microperforated_impedance(
        diameter, thickness, porosity, omega, air
    )

    cell = _plate_matrix(plate_impedance[:, :, 0, :])
    cell = cell @ _cavity_matrix(wavenumber, characteristic_impedance, depth[:, :, 0])
    cell = cell @ _plate_matrix(plate_impedance[:, :, 1, :])
    cell = cell @ _cavity_matrix(wavenumber, characteristic_impedance, depth[:, :, 1])
    cell = cell @ _plate_matrix(plate_impedance[:, :, 2, :])

    matrix_a = cell[..., 0, 0]
    matrix_b = cell[..., 0, 1]
    matrix_c = cell[..., 1, 0]
    matrix_d = cell[..., 1, 1]
    admittance = np.empty_like(cell)
    admittance[..., 0, 0] = matrix_d / matrix_b
    admittance[..., 0, 1] = matrix_c - matrix_a * matrix_d / matrix_b
    admittance[..., 1, 0] = 1.0 / matrix_b
    admittance[..., 1, 1] = -matrix_a / matrix_b

    # 原脚本中四个单胞截面相同，权重都是 1/4。
    weights = np.full(diameter.shape[1], 1.0 / diameter.shape[1], dtype=np.float64)
    parallel = np.sum(weights[None, :, None, None, None] * admittance, axis=1)

    y11 = parallel[..., 0, 0]
    y12 = parallel[..., 0, 1]
    y21 = parallel[..., 1, 0]
    y22 = parallel[..., 1, 1]
    matrix_a = -y22 / y21
    matrix_b = 1.0 / y21
    matrix_c = y12 - y11 * y22 / y21
    matrix_d = y11 / y21

    z0 = air.impedance
    denominator = matrix_a + matrix_b / z0 + matrix_c * z0 + matrix_d
    transmission = 2.0 / denominator
    reflection_forward = (
        matrix_a + matrix_b / z0 - matrix_c * z0 - matrix_d
    ) / denominator
    reflection_backward = (
        -matrix_a + matrix_b / z0 - matrix_c * z0 + matrix_d
    ) / denominator

    reflectance_forward = np.abs(reflection_forward) ** 2
    reflectance_backward = np.abs(reflection_backward) ** 2
    transmittance = np.abs(transmission) ** 2
    # 特征值间隔等于 2 |sqrt(rf * rb)|，这里保留与原脚本相同的写法。
    splitting = np.sqrt(reflection_forward * reflection_backward)
    eigenvalue_gap = np.abs((transmission + splitting) - (transmission - splitting))

    result = {
        "f": frequency,
        "alpha_f": 1.0 - reflectance_forward - transmittance,
        "alpha_b": 1.0 - reflectance_backward - transmittance,
        "Rf": reflectance_forward,
        "Rb": reflectance_backward,
        "Tr": transmittance,
        "ep_diff": eigenvalue_gap,
        "tau": transmission,
        "rf": reflection_forward,
        "rb": reflection_backward,
        "cell_transfer": cell,
        "transfer_a": matrix_a,
        "transfer_b": matrix_b,
        "transfer_c": matrix_c,
        "transfer_d": matrix_d,
    }
    if single:
        for key, value in result.items():
            if key == "f":
                continue
            result[key] = value[0]
    return result


def identical_cell_error(
    diameter_row: np.ndarray | None = None,
    frequency: np.ndarray | None = None,
) -> float:
    """四个完全相同的单胞并联后，总传递矩阵应回到单胞矩阵。"""

    if frequency is None:
        frequency = frequency_axis()
    nominal_d, nominal_t, nominal_h = nominal_geometry()
    if diameter_row is None:
        diameter_row = nominal_d[0]
    diameter = np.repeat(diameter_row[None, :], N_CELLS, axis=0)
    thickness = np.repeat(nominal_t[0][None, :], N_CELLS, axis=0)
    depth = np.repeat(nominal_h[0][None, :], N_CELLS, axis=0)
    response = transfer_response(diameter, thickness, depth, frequency)
    total = np.stack(
        [
            np.stack([response["transfer_a"], response["transfer_b"]], axis=-1),
            np.stack([response["transfer_c"], response["transfer_d"]], axis=-1),
        ],
        axis=-2,
    )
    cell = response["cell_transfer"][0]
    return float(np.max(np.abs(total - cell)))


def sample_geometries(
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在训练边界内均匀抽取几何。"""

    diameter = rng.uniform(D_MIN_M, D_MAX_M, size=(count, N_CELLS, N_PLATES))
    thickness = rng.uniform(T_MIN_M, T_MAX_M, size=(count, N_CELLS, N_PLATES))
    depth = rng.uniform(H_MIN_M, H_MAX_M, size=(count, N_CELLS, N_CAVITIES))
    return diameter, thickness, depth


def spectra_targets(response: dict[str, np.ndarray]) -> np.ndarray:
    """网络监督信号，形状 ``(batch, 3, frequency)``。"""

    return np.stack([response["alpha_f"], response["alpha_b"], response["ep_diff"]], axis=-2)
