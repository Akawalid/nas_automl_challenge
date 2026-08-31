from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoisyPreActivitiesGradSettings:
    enabled: bool = False
    std: float = 0.0
    relative: bool = True


@dataclass(frozen=True)
class VarianceTransferSettings:
    enabled: bool = False
    rescaling: str | None = None


def _normalise_optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalised = str(value).strip()
    if normalised in ("", "none", "null"):
        return None
    return normalised


def get_noisy_pre_activities_grad_settings(
    cfg: dict,
) -> NoisyPreActivitiesGradSettings:
    growth_cfg = cfg.get("growth", {})
    if not isinstance(growth_cfg, dict):
        return NoisyPreActivitiesGradSettings()

    ablations = growth_cfg.get("ablations", {})
    if not isinstance(ablations, dict):
        raise ValueError("growth.ablations must be a mapping")

    noisy_grad = ablations.get("noisy_pre_activities_grad", {})
    if not isinstance(noisy_grad, dict):
        raise ValueError("growth.ablations.noisy_pre_activities_grad must be a mapping")

    enabled = noisy_grad.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "growth.ablations.noisy_pre_activities_grad.enabled must be a boolean"
        )
    if not enabled:
        return NoisyPreActivitiesGradSettings()

    std = noisy_grad.get("std")
    if isinstance(std, bool) or not isinstance(std, (int, float)) or std <= 0:
        raise ValueError(
            "growth.ablations.noisy_pre_activities_grad.std must be a "
            "positive number when enabled"
        )

    relative = noisy_grad.get("relative", True)
    if not isinstance(relative, bool):
        raise ValueError(
            "growth.ablations.noisy_pre_activities_grad.relative must be a boolean"
        )

    return NoisyPreActivitiesGradSettings(
        enabled=True,
        std=float(std),
        relative=relative,
    )


def get_variance_transfer_settings(cfg: dict) -> VarianceTransferSettings:
    growth_cfg = cfg.get("growth", {})
    if not isinstance(growth_cfg, dict):
        return VarianceTransferSettings()

    ablations = growth_cfg.get("ablations", {})
    if not isinstance(ablations, dict):
        raise ValueError("growth.ablations must be a mapping")

    variance_transfer = ablations.get("variance_transfer", {})
    if not isinstance(variance_transfer, dict):
        raise ValueError("growth.ablations.variance_transfer must be a mapping")

    enabled = variance_transfer.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("growth.ablations.variance_transfer.enabled must be a boolean")
    if not enabled:
        return VarianceTransferSettings()

    rescaling = _normalise_optional_string(
        variance_transfer.get("rescaling", "vt_constraint_old_shape")
    )
    valid_rescalings = {
        "default_vt",
        "vt_constraint_old_shape",
        "vt_constraint_new_shape",
    }
    if rescaling not in valid_rescalings:
        raise ValueError(
            "growth.ablations.variance_transfer.rescaling must be one of: "
            "default_vt, vt_constraint_old_shape, vt_constraint_new_shape"
        )

    return VarianceTransferSettings(enabled=True, rescaling=rescaling)


def add_noise_to_pre_activities_grad(
    pre_activities_grad: dict[str, torch.Tensor],
    cfg: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    settings = get_noisy_pre_activities_grad_settings(cfg)
    if not settings.enabled:
        return pre_activities_grad, {}

    noisy_pre_activities_grad = {}
    gradient_rms_values = []
    noise_rms_values = []

    for name, grad in pre_activities_grad.items():
        if grad.numel() == 0 or not torch.is_floating_point(grad):
            noisy_pre_activities_grad[name] = grad
            continue

        if settings.relative:
            scale = grad.std(unbiased=False)
        else:
            scale = torch.ones((), dtype=grad.dtype, device=grad.device)

        noise = torch.randn_like(grad) * (settings.std * scale)
        noisy_pre_activities_grad[name] = grad + noise

        gradient_rms_values.append(
            torch.linalg.vector_norm(grad.detach()) / grad.numel() ** 0.5
        )
        noise_rms_values.append(
            torch.linalg.vector_norm(noise.detach()) / noise.numel() ** 0.5
        )

    metrics = {
        "std": settings.std,
        "relative": float(settings.relative),
    }
    if gradient_rms_values:
        metrics["gradient_rms_mean"] = float(torch.stack(gradient_rms_values).mean())
        metrics["noise_rms_mean"] = float(torch.stack(noise_rms_values).mean())

    return noisy_pre_activities_grad, metrics
