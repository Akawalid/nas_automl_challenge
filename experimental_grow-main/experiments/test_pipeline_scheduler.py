import pytest
import torch

from experiments.schedulers import (
    get_pipeline_linear_warmup_settings,
    get_pipeline_scheduler,
    get_pipeline_scheduler_ablation_settings,
    get_pipeline_warmup_constant_scheduler,
    get_persistent_pipeline_scheduler,
    is_pipeline_linear_warmup_active,
    rebind_pipeline_scheduler,
    rebind_pipeline_scheduler_with_warmup,
    should_defer_pipeline_scheduler,
)
from tools.lr_scheduler import WarmupCosineAnnealingLR


def _optimizer(lr: float = 0.1) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.SGD([parameter], lr=lr)


def test_disabled_ablation_parameters_are_ignored() -> None:
    settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "linear_warmup": {"enabled": False, "epochs": "unused"},
                "global_scheduler": {
                    "enabled": False,
                    "total_epochs": "unused",
                    "rebind_linear_warmup": {
                        "enabled": False,
                        "epochs": "unused",
                    },
                },
                "post_growth_scheduler": {
                    "enabled": False,
                    "growth_linear_warmup": {
                        "enabled": False,
                        "epochs": "unused",
                    },
                },
            }
        }
    )

    assert settings.linear_warmup_enabled is False
    assert settings.global_scheduler_enabled is False
    assert settings.global_rebind_linear_warmup_enabled is False
    assert settings.post_growth_scheduler_enabled is False
    assert settings.growth_linear_warmup_enabled is False


def test_disabled_linear_warmup_preserves_cosine_scheduler() -> None:
    actual_optimizer = _optimizer()
    expected_optimizer = _optimizer()

    actual = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=actual_optimizer,
        num_epochs=5,
        eta_min=0.01,
        linear_warmup=False,
        warmup_epochs=2,
        clamp_after_end=False,
    )
    expected = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=expected_optimizer,
        T_max=5,
        eta_min=0.01,
    )

    assert type(actual) is type(expected)
    for _ in range(6):
        assert actual_optimizer.param_groups[0]["lr"] == pytest.approx(
            expected_optimizer.param_groups[0]["lr"]
        )
        actual_optimizer.step()
        expected_optimizer.step()
        actual.step()
        expected.step()


def test_enabled_linear_warmup_precedes_cosine_scheduler() -> None:
    optimizer = _optimizer()
    scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=optimizer,
        num_epochs=5,
        eta_min=0.01,
        linear_warmup=True,
        warmup_epochs=2,
        clamp_after_end=False,
    )

    assert isinstance(scheduler, WarmupCosineAnnealingLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.055)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_early_stopping_is_suspended_only_during_enabled_warmup() -> None:
    optimizer = _optimizer()
    scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=optimizer,
        num_epochs=5,
        eta_min=0.01,
        linear_warmup=True,
        warmup_epochs=2,
        clamp_after_end=False,
    )
    enabled_settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "linear_warmup": {"enabled": True, "epochs": 2},
            }
        }
    )
    disabled_settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "linear_warmup": {"enabled": False, "epochs": 2},
            }
        }
    )

    assert is_pipeline_linear_warmup_active(scheduler, enabled_settings)
    assert not is_pipeline_linear_warmup_active(scheduler, disabled_settings)

    for _ in range(2):
        optimizer.step()
        scheduler.step()

    assert not is_pipeline_linear_warmup_active(scheduler, enabled_settings)


def test_post_growth_scheduler_defers_only_until_growth_completes() -> None:
    settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "post_growth_scheduler": {"enabled": True},
            }
        }
    )
    disabled_settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "post_growth_scheduler": {"enabled": False},
            }
        }
    )

    assert should_defer_pipeline_scheduler(settings, growth_complete=False)
    assert not should_defer_pipeline_scheduler(settings, growth_complete=True)
    assert not should_defer_pipeline_scheduler(
        disabled_settings,
        growth_complete=False,
    )


@pytest.mark.parametrize("enabled", [None, "true", 1])
def test_post_growth_scheduler_enabled_must_be_boolean(enabled) -> None:
    with pytest.raises(
        ValueError,
        match=r"training\.ablations\.post_growth_scheduler\.enabled",
    ):
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "post_growth_scheduler": {"enabled": enabled},
                }
            }
        )


def test_post_growth_scheduler_cannot_use_top_level_linear_warmup() -> None:
    with pytest.raises(
        ValueError,
        match=r"post_growth_scheduler cannot be combined",
    ):
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "post_growth_scheduler": {"enabled": True},
                    "linear_warmup": {"enabled": True, "epochs": 2},
                }
            }
        )


def test_post_growth_scheduler_can_be_combined_with_global_scheduler() -> None:
    settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "post_growth_scheduler": {"enabled": True},
                "global_scheduler": {"enabled": True, "total_epochs": 350},
            }
        }
    )

    assert settings.post_growth_scheduler_enabled is True
    assert settings.global_scheduler_enabled is True
    assert settings.global_scheduler_total_epochs == 350


def test_global_scheduler_can_enable_rebind_linear_warmup() -> None:
    settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "global_scheduler": {
                    "enabled": True,
                    "total_epochs": 350,
                    "rebind_linear_warmup": {"enabled": True, "epochs": 4},
                },
            }
        }
    )

    assert settings.global_rebind_linear_warmup_enabled is True
    assert settings.global_rebind_linear_warmup_epochs == 4


def test_global_rebind_linear_warmup_requires_global_scheduler() -> None:
    with pytest.raises(
        ValueError,
        match=r"global_scheduler\.rebind_linear_warmup requires",
    ):
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "global_scheduler": {
                        "enabled": False,
                        "rebind_linear_warmup": {"enabled": True, "epochs": 4},
                    },
                }
            }
        )


@pytest.mark.parametrize("warmup_epochs", [None, 0, "4"])
def test_global_rebind_linear_warmup_requires_valid_epoch_count(
    warmup_epochs,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"global_scheduler\.rebind_linear_warmup\.epochs",
    ):
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "global_scheduler": {
                        "enabled": True,
                        "total_epochs": 350,
                        "rebind_linear_warmup": {
                            "enabled": True,
                            "epochs": warmup_epochs,
                        },
                    },
                }
            }
        )


def test_post_growth_scheduler_can_enable_growth_linear_warmup() -> None:
    settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "post_growth_scheduler": {
                    "enabled": True,
                    "growth_linear_warmup": {"enabled": True, "epochs": 5},
                },
            }
        }
    )

    assert settings.growth_linear_warmup_enabled is True
    assert settings.growth_linear_warmup_epochs == 5
    assert get_pipeline_linear_warmup_settings(
        settings,
        growth_complete=False,
    ) == (True, 5)
    assert get_pipeline_linear_warmup_settings(
        settings,
        growth_complete=True,
    ) == (False, None)


def test_growth_linear_warmup_requires_post_growth_scheduler() -> None:
    with pytest.raises(
        ValueError,
        match=r"post_growth_scheduler\.growth_linear_warmup requires",
    ):
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "post_growth_scheduler": {
                        "enabled": False,
                        "growth_linear_warmup": {"enabled": True, "epochs": 5},
                    },
                }
            }
        )


@pytest.mark.parametrize("warmup_epochs", [None, 0, "5"])
def test_growth_linear_warmup_requires_valid_epoch_count(
    warmup_epochs,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"post_growth_scheduler\.growth_linear_warmup\.epochs",
    ):
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "post_growth_scheduler": {
                        "enabled": True,
                        "growth_linear_warmup": {
                            "enabled": True,
                            "epochs": warmup_epochs,
                        },
                    },
                }
            }
        )


@pytest.mark.parametrize("warmup_epochs", [None, 0, 5])
def test_enabled_linear_warmup_requires_valid_epoch_count(warmup_epochs) -> None:
    with pytest.raises(
        ValueError,
        match=r"training\.ablations\.linear_warmup\.epochs",
    ):
        get_pipeline_scheduler(
            scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
            optimizer=_optimizer(),
            num_epochs=5,
            eta_min=0.01,
            linear_warmup=True,
            warmup_epochs=warmup_epochs,
            clamp_after_end=False,
        )


def test_growth_linear_warmup_uses_warmup_only_during_growth() -> None:
    optimizer = _optimizer()
    scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=optimizer,
        num_epochs=6,
        eta_min=0.01,
        linear_warmup=True,
        warmup_epochs=2,
        clamp_after_end=False,
    )
    settings = get_pipeline_scheduler_ablation_settings(
        {
            "ablations": {
                "post_growth_scheduler": {
                    "enabled": True,
                    "growth_linear_warmup": {"enabled": True, "epochs": 2},
                },
            }
        }
    )

    assert is_pipeline_linear_warmup_active(
        scheduler,
        settings,
        growth_complete=False,
    )
    assert not is_pipeline_linear_warmup_active(
        scheduler,
        settings,
        growth_complete=True,
    )

    for _ in range(2):
        optimizer.step()
        scheduler.step()

    assert not is_pipeline_linear_warmup_active(
        scheduler,
        settings,
        growth_complete=False,
    )


def test_growth_linear_warmup_scheduler_reaches_and_keeps_base_lr() -> None:
    optimizer = _optimizer(lr=0.1)
    scheduler = get_pipeline_warmup_constant_scheduler(
        optimizer=optimizer,
        warmup_epochs=2,
        warmup_start_lr=0.01,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.055)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_global_scheduler_continues_after_optimizer_recreation() -> None:
    uninterrupted_optimizer = _optimizer()
    uninterrupted_scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=uninterrupted_optimizer,
        num_epochs=5,
        eta_min=0.01,
        linear_warmup=False,
        clamp_after_end=True,
    )

    first_optimizer = _optimizer()
    persistent_scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=first_optimizer,
        num_epochs=5,
        eta_min=0.01,
        linear_warmup=False,
        clamp_after_end=True,
    )

    for _ in range(2):
        uninterrupted_optimizer.step()
        uninterrupted_scheduler.step()
        first_optimizer.step()
        persistent_scheduler.step()

    second_optimizer = _optimizer()
    rebound_scheduler = rebind_pipeline_scheduler(
        persistent_scheduler,
        second_optimizer,
    )

    assert rebound_scheduler is persistent_scheduler
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted_optimizer.param_groups[0]["lr"]
    )

    for _ in range(3):
        uninterrupted_optimizer.step()
        uninterrupted_scheduler.step()
        second_optimizer.step()
        rebound_scheduler.step()
        assert second_optimizer.param_groups[0]["lr"] == pytest.approx(
            uninterrupted_optimizer.param_groups[0]["lr"]
        )


def test_global_scheduler_does_not_restart_linear_warmup() -> None:
    first_optimizer = _optimizer()
    scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=first_optimizer,
        num_epochs=6,
        eta_min=0.01,
        linear_warmup=True,
        warmup_epochs=2,
        clamp_after_end=True,
    )

    first_optimizer.step()
    scheduler.step()
    assert first_optimizer.param_groups[0]["lr"] == pytest.approx(0.055)

    second_optimizer = _optimizer()
    rebind_pipeline_scheduler(scheduler, second_optimizer)
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(0.055)

    second_optimizer.step()
    scheduler.step()
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_global_scheduler_rebind_can_warm_up_to_current_cosine_lr() -> None:
    first_optimizer = _optimizer()
    scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=first_optimizer,
        num_epochs=6,
        eta_min=0.01,
        linear_warmup=False,
        clamp_after_end=True,
    )

    first_optimizer.step()
    scheduler.step()
    target_lr = scheduler.get_last_lr()[0]

    second_optimizer = _optimizer()
    warmup_scheduler = rebind_pipeline_scheduler_with_warmup(
        scheduler,
        second_optimizer,
        warmup_epochs=2,
        warmup_start_lr=0.01,
    )

    assert get_persistent_pipeline_scheduler(warmup_scheduler) is scheduler
    assert is_pipeline_linear_warmup_active(
        warmup_scheduler,
        get_pipeline_scheduler_ablation_settings(
            {
                "ablations": {
                    "global_scheduler": {
                        "enabled": True,
                        "total_epochs": 6,
                        "rebind_linear_warmup": {
                            "enabled": True,
                            "epochs": 2,
                        },
                    },
                }
            }
        ),
    )
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    second_optimizer.step()
    warmup_scheduler.step()
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(
        0.01 + 0.5 * (target_lr - 0.01)
    )

    second_optimizer.step()
    warmup_scheduler.step()
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(target_lr)

    previous_scheduler_epoch = scheduler.last_epoch
    second_optimizer.step()
    warmup_scheduler.step()
    assert scheduler.last_epoch == previous_scheduler_epoch + 1


def test_global_scheduler_stays_at_eta_min_after_total_epochs() -> None:
    optimizer = _optimizer()
    scheduler = get_pipeline_scheduler(
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=optimizer,
        num_epochs=3,
        eta_min=0.01,
        linear_warmup=False,
        clamp_after_end=True,
    )

    for _ in range(6):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
