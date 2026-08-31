import pytest
import torch

from experiments.growth_ablations import (
    add_noise_to_pre_activities_grad,
    get_noisy_pre_activities_grad_settings,
    get_variance_transfer_settings,
)


def test_disabled_noisy_pre_activities_grad_returns_original_tensors() -> None:
    gradients = {"a": torch.arange(4, dtype=torch.float32)}

    result, metrics = add_noise_to_pre_activities_grad(
        gradients,
        {
            "growth": {
                "ablations": {
                    "noisy_pre_activities_grad": {
                        "enabled": False,
                        "std": "ignored",
                    }
                }
            }
        },
    )

    assert result is gradients
    assert metrics == {}


def test_relative_noisy_pre_activities_grad_adds_noise() -> None:
    torch.manual_seed(0)
    gradients = {"a": torch.arange(6, dtype=torch.float32)}

    result, metrics = add_noise_to_pre_activities_grad(
        gradients,
        {
            "growth": {
                "ablations": {
                    "noisy_pre_activities_grad": {
                        "enabled": True,
                        "std": 0.1,
                        "relative": True,
                    }
                }
            }
        },
    )

    assert result is not gradients
    assert result["a"].shape == gradients["a"].shape
    assert not torch.equal(result["a"], gradients["a"])
    assert metrics["std"] == pytest.approx(0.1)
    assert metrics["relative"] == pytest.approx(1.0)
    assert metrics["noise_rms_mean"] > 0


def test_absolute_noisy_pre_activities_grad_can_perturb_zero_tensor() -> None:
    torch.manual_seed(0)
    gradients = {"a": torch.zeros(6)}

    result, _ = add_noise_to_pre_activities_grad(
        gradients,
        {
            "growth": {
                "ablations": {
                    "noisy_pre_activities_grad": {
                        "enabled": True,
                        "std": 0.1,
                        "relative": False,
                    }
                }
            }
        },
    )

    assert not torch.equal(result["a"], gradients["a"])


@pytest.mark.parametrize("std", [None, 0, -0.1, True])
def test_enabled_noisy_pre_activities_grad_requires_positive_std(std) -> None:
    with pytest.raises(
        ValueError,
        match=r"growth\.ablations\.noisy_pre_activities_grad\.std",
    ):
        get_noisy_pre_activities_grad_settings(
            {
                "growth": {
                    "ablations": {
                        "noisy_pre_activities_grad": {
                            "enabled": True,
                            "std": std,
                        }
                    }
                }
            }
        )


def test_disabled_variance_transfer_returns_noop_settings() -> None:
    settings = get_variance_transfer_settings(
        {
            "growth": {
                "ablations": {
                    "variance_transfer": {
                        "enabled": False,
                        "rescaling": "not_used",
                    }
                }
            }
        }
    )

    assert not settings.enabled
    assert settings.rescaling is None


def test_enabled_variance_transfer_defaults_to_old_shape_constraint() -> None:
    settings = get_variance_transfer_settings(
        {"growth": {"ablations": {"variance_transfer": {"enabled": True}}}}
    )

    assert settings.enabled
    assert settings.rescaling == "vt_constraint_old_shape"


def test_enabled_variance_transfer_validates_rescaling() -> None:
    with pytest.raises(
        ValueError,
        match=r"growth\.ablations\.variance_transfer\.rescaling",
    ):
        get_variance_transfer_settings(
            {
                "growth": {
                    "ablations": {
                        "variance_transfer": {
                            "enabled": True,
                            "rescaling": "unsupported",
                        }
                    }
                }
            }
        )
