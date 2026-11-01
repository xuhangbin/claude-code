"""Checks for the transfer-matrix forward model."""

import numpy as np

from origami_mpp_tandem.physics import absorption, equivalent_height, example_params, frequency_axis, stinson_fluid


def test_full_height_is_not_reduced():
    assert np.isclose(equivalent_height(np.array([0.08])), 0.08)


def test_example_absorption_is_physical():
    freq = frequency_axis()
    alpha = absorption(example_params(), freq, fluid=stinson_fluid(freq))
    assert alpha.shape == (601,)
    assert np.isfinite(alpha).all()
    assert alpha.min() >= -1e-6
    assert alpha.max() <= 1 + 1e-6
