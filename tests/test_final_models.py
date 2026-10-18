"""Contracts for the actual selected model preprocessing and ranking logic."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from distress_v2.preprocessing import NumericTransform
from distress_v2.training import sample_weights
from distress_v2.finalize import annual_ranks
from distress_v2.uncertainty import contract
from distress_v2.networks import NumericNetwork, probabilities
from distress_v2.data import features
import distress_v2.data as data_module


@pytest.mark.parametrize("kind", ["native", "std01", "quantile"])
def test_selected_transform_does_not_refit_or_distort_masks(kind):
    x = pd.DataFrame({"amount": np.linspace(-10, 10, 64), "mask": [0, 1] * 32})
    transform = NumericTransform(kind, indicators=False).fit(x)
    median = transform.median.copy()
    z = transform.transform(x)
    assert set(z[:, -1]) == {0, 1}
    transform.transform(x.assign(amount=1e12))
    np.testing.assert_array_equal(transform.median, median)


@pytest.mark.parametrize("beta", [0.5, 1])
def test_company_weighting_accepts_integer_counts(beta):
    df = pd.DataFrame({"company_name": ["a", "a", "b"], "year": [2012, 2013, 2013], "y": [0, 0, 1]})
    weight = sample_weights(df, {"company_beta": beta})
    assert np.isfinite(weight).all() and (weight > 0).all()
    np.testing.assert_allclose(weight[0] / weight[2], 2 ** (-beta))


def test_three_year_half_life_uses_training_cutoff():
    df = pd.DataFrame({"year": [2008, 2011, 2014], "y": [0, 1, 0]})
    weight = sample_weights(df, {"half_life": 3})
    np.testing.assert_allclose(weight / weight[-1], [0.25, 0.5, 1])


def test_rank_blend_is_invariant_to_other_year_scale():
    df = pd.DataFrame({"year": [2015, 2015, 2016, 2016]})
    score = np.array([0.1, 0.7, 0.2, 0.9])
    a = annual_ranks(df, score)
    score[2:] *= 1000
    np.testing.assert_array_equal(a[:2], annual_ranks(df, score)[:2])


def test_paired_bootstrap_matches_weighted_reference_and_fixed_years():
    assert contract()["passed"]


def test_final_mlp_optimizer_and_serialization(tmp_path):
    torch.manual_seed(42)
    x = np.random.default_rng(42).normal(size=(24, 17)).astype("float32")
    config = {"family": "mlp", "width": 128, "depth": 2, "dropout": 0.15}
    model = NumericNetwork(17, 17, config, x_for_bins=x)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(x)), torch.tensor([0., 1.] * 12))
    loss.backward()
    assert torch.isfinite(loss)
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    path = tmp_path / "fixture.pt"
    torch.save(model.state_dict(), path)
    restored = NumericNetwork(17, 17, config, x_for_bins=x)
    restored.load_state_dict(torch.load(path, weights_only=True))
    np.testing.assert_allclose(probabilities(model, x), probabilities(restored, x), atol=1e-7)


@pytest.mark.parametrize("feature_set", ["RAW17", "LEGACY33", "LAGS_SLOG3"])
def test_relocated_feature_builders_exclude_labels_and_future(feature_set):
    df = pd.DataFrame([(f"firm_{i}", y) for i in range(5) for y in range(2010, 2015)],
                      columns=["company_name", "year"])
    for k in range(1, 19):
        df[f"X{k}"] = np.random.default_rng(k).uniform(1, 100, len(df))
    df["X16"] = df.X9
    df["row_id"] = np.arange(len(df))
    df["y"] = np.arange(len(df)) % 2
    data_module._features.clear()
    original = features(df, feature_set).copy()
    other = df.copy()
    other.loc[other.year == 2014, "X6"] *= 1e9
    data_module._features.clear()
    changed = features(other, feature_set)
    pd.testing.assert_frame_equal(original[df.year < 2014], changed[df.year < 2014])
    assert not {"company_name", "year", "y", "row_id"} & set(original)
    data_module._features.clear()
