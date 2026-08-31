import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.functional_gradient import (
    capture_network_outputs,
    measure_output_functional_metrics,
    prepare_fixed_train_probe,
)


def _identity_classifier() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    return model


def test_fixed_train_probe_preserves_the_original_epoch_batches():
    inputs = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    targets = torch.arange(12)
    dataset = TensorDataset(inputs, targets)
    diagnostic_loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )
    expected_loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )

    probe, preserved_epoch = prepare_fixed_train_probe(
        diagnostic_loader,
        num_batches=2,
    )
    actual_batches = list(preserved_epoch)
    expected_batches = list(expected_loader)

    assert len(actual_batches) == len(expected_batches)
    for actual, expected in zip(actual_batches, expected_batches):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    for probe_batch, expected_batch in zip(probe, expected_batches[:2]):
        torch.testing.assert_close(probe_batch[0], expected_batch[0])
        torch.testing.assert_close(probe_batch[1], expected_batch[1])


def test_squared_l2_norm_matches_output_functional_gradient():
    inputs = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.7], [0.8, -0.1]])
    targets = torch.tensor([0, 1, 1, 0])
    model = _identity_classifier()
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    measured = measure_output_functional_metrics(
        model,
        dataloader,
        torch.nn.CrossEntropyLoss(reduction="sum"),
        "cpu",
    )
    expected_gradient = (
        torch.softmax(inputs, dim=1)
        - torch.nn.functional.one_hot(targets, num_classes=2)
    )
    expected = expected_gradient.square().sum() / len(inputs)

    torch.testing.assert_close(
        torch.tensor(measured.gradient_squared_l2_norm), expected
    )
    assert measured.update_squared_l2_norm is None
    assert measured.scale_optimal_learning_rate is None
    assert measured.approximation_l2_distance is None
    assert measured.relative_error_approximation_denominator is None
    assert measured.relative_error_gradient_denominator is None
    assert measured.directional_cosine is None


def test_squared_l2_norm_is_independent_of_batch_size_and_restores_mode():
    inputs = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.7], [0.8, -0.1]])
    targets = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(inputs, targets)
    model = _identity_classifier()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    batch_size_one = measure_output_functional_metrics(
        model, DataLoader(dataset, batch_size=1), loss_fn, "cpu"
    )
    full_batch = measure_output_functional_metrics(
        model, DataLoader(dataset, batch_size=len(dataset)), loss_fn, "cpu"
    )

    assert (
        batch_size_one.gradient_squared_l2_norm
        == full_batch.gradient_squared_l2_norm
    )
    assert model.training is True


def test_signed_eta_is_reported_for_an_anti_aligned_update():
    inputs = torch.tensor([0, 1])
    targets = torch.tensor([0, 1])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    model = torch.nn.Embedding(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    reference_outputs = capture_network_outputs(model, dataloader, "cpu")
    reference_gradient = (
        torch.softmax(reference_outputs, dim=1)
        - torch.nn.functional.one_hot(targets, num_classes=2)
    )
    with torch.no_grad():
        model.weight.add_(reference_gradient)

    measured = measure_output_functional_metrics(
        model,
        dataloader,
        torch.nn.CrossEntropyLoss(reduction="sum"),
        "cpu",
        reference_outputs=reference_outputs,
    )

    torch.testing.assert_close(
        torch.tensor(measured.scale_optimal_learning_rate),
        torch.tensor(-1.0),
    )
    torch.testing.assert_close(
        torch.tensor(measured.directional_cosine),
        torch.tensor(-1.0),
    )
    assert measured.approximation_l2_distance is None
    assert measured.relative_error_approximation_denominator is None
    assert measured.relative_error_gradient_denominator is None


def test_output_update_is_the_realized_finite_displacement():
    inputs = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.7], [0.8, -0.1]])
    targets = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2)
    model = _identity_classifier()
    reference_outputs = capture_network_outputs(model, dataloader, "cpu")

    with torch.no_grad():
        model.weight.add_(torch.tensor([[0.1, -0.2], [-0.1, 0.2]]))

    measured = measure_output_functional_metrics(
        model,
        dataloader,
        torch.nn.CrossEntropyLoss(reduction="sum"),
        "cpu",
        reference_outputs=reference_outputs,
    )
    current_outputs = model(inputs).detach()
    expected = (reference_outputs - current_outputs).square().sum() / len(inputs)
    expected_reference_gradient = (
        torch.softmax(reference_outputs, dim=1)
        - torch.nn.functional.one_hot(targets, num_classes=2)
    )
    expected_gradient_norm = (
        expected_reference_gradient.square().sum() / len(inputs)
    )
    functional_update = reference_outputs - current_outputs
    dot_product = torch.sum(functional_update * expected_reference_gradient) / len(
        inputs
    )
    update_squared_norm = functional_update.square().sum() / len(inputs)
    eta_star = dot_product / expected_gradient_norm
    approximation = functional_update / eta_star
    approximation_error = approximation - expected_reference_gradient
    distance = torch.sqrt(approximation_error.square().sum() / len(inputs))
    approximation_norm = torch.sqrt(approximation.square().sum() / len(inputs))
    gradient_norm = torch.sqrt(expected_gradient_norm)
    cosine = dot_product / torch.sqrt(update_squared_norm * expected_gradient_norm)

    torch.testing.assert_close(
        torch.tensor(measured.update_squared_l2_norm), expected
    )
    torch.testing.assert_close(
        torch.tensor(measured.gradient_squared_l2_norm), expected_gradient_norm
    )
    torch.testing.assert_close(
        torch.tensor(measured.scale_optimal_learning_rate), eta_star
    )
    torch.testing.assert_close(
        torch.tensor(measured.approximation_l2_distance), distance
    )
    torch.testing.assert_close(
        torch.tensor(measured.relative_error_approximation_denominator),
        distance / approximation_norm,
    )
    torch.testing.assert_close(
        torch.tensor(measured.relative_error_gradient_denominator),
        distance / gradient_norm,
    )
    torch.testing.assert_close(torch.tensor(measured.directional_cosine), cosine)
    torch.testing.assert_close(measured.output_snapshot, current_outputs)
