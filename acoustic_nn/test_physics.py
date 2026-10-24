"""传递矩阵移植的回归检查。直接运行：python -m acoustic_nn.test_physics"""

from __future__ import annotations

import numpy as np

from acoustic_nn.physics import (
    frequency_axis,
    identical_cell_error,
    nominal_geometry,
    pack_geometry,
    transfer_response,
    unpack_geometry,
)


def main() -> None:
    frequency = frequency_axis()
    assert frequency.size == 401
    assert frequency[0] == 100.0 and frequency[-1] == 500.0

    error = identical_cell_error(frequency=frequency)
    if error >= 1e-8:
        raise AssertionError(f"相同单胞并联不一致，最大误差 {error:.3e}")

    diameter, thickness, depth = nominal_geometry()
    restored_d, restored_t, restored_h = unpack_geometry(
        pack_geometry(diameter, thickness, depth)
    )
    if not (
        np.allclose(restored_d, diameter)
        and np.allclose(restored_t, thickness)
        and np.allclose(restored_h, depth)
    ):
        raise AssertionError("几何打包往返失败")

    response = transfer_response(diameter, thickness, depth, frequency)
    determinant = (
        response["transfer_a"] * response["transfer_d"]
        - response["transfer_b"] * response["transfer_c"]
    )
    if np.max(np.abs(determinant - 1.0)) > 1e-8:
        raise AssertionError("总传递矩阵不满足互易条件 det=1")
    energy = response["alpha_f"] + response["Rf"] + response["Tr"]
    if np.max(np.abs(energy - 1.0)) > 1e-8:
        raise AssertionError("前向能量不守恒")
    for name in ("alpha_f", "alpha_b", "Rf", "Rb", "Tr", "ep_diff"):
        values = response[name]
        if not np.isfinite(values).all():
            raise AssertionError(f"{name} 出现非有限值")
        if np.min(values) < -1e-8 or (name != "ep_diff" and np.max(values) > 1.0 + 1e-6):
            raise AssertionError(f"{name} 超出被动范围")
    print(f"physics ok  identical_cell_error={error:.3e}")
    print(
        f"nominal mean alpha_f={response['alpha_f'].mean():.4f}  "
        f"alpha_b={response['alpha_b'].mean():.4f}  "
        f"ep_diff={response['ep_diff'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
