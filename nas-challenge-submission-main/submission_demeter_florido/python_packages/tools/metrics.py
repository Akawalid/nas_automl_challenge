import logging

import torch
from deprecated import deprecated
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryFBetaScore,
    BinaryPrecisionRecallCurve,
)
from torchmetrics.utilities.compute import auc


@deprecated(reason="Use the TopKAccuracy class instead.")
def topk_accuracy(y_pred, y, k=1):
    result = y_pred.topk(k, dim=1).indices == y.unsqueeze(1)
    return result.sum() / y.size(0)


# def sk_balanced_topk_accuracy(y_pred, y, k=1):
#     result = balanced_accuracy_score(
#         y.unsqueeze(1).detach().cpu().numpy(),
#         (y_pred.topk(k, dim=1).indices).detach().cpu().numpy(),
#     )
#     return result
#
#
# def balanced_topk_accuracy(
#     y_pred: torch.Tensor, y: torch.Tensor, n_class: int, k=1
# ) -> torch.Tensor:
#     """Compute the balanced top-k accuracy.
#
#     Parameters
#     ----------
#     y_pred : torch.Tensor
#         The predicted logits or probabilities of shape (batch_size, num_classes).
#     y : torch.Tensor
#         The true labels of shape (batch_size,).
#     n_class : int
#         The number of classes.
#     k : int
#         The 'k' in top-k accuracy.
#
#     Returns
#     -------
#     torch.Tensor
#         The balanced top-k accuracy as a tensor.
#     """
#     # Check if true label is in top-k predictions
#     top_k_preds = y_pred.topk(k, dim=1).indices  # (batch_size, k)
#     correct = (top_k_preds == y.unsqueeze(1)).any(dim=1)  # (batch_size,)
#
#     # Compute per-class recall and average (balanced accuracy)
#     per_class_recall = torch.zeros(n_class, device=y_pred.device)
#     for c in range(n_class):
#         class_mask = y == c
#         if class_mask.sum() > 0:
#             per_class_recall[c] = correct[class_mask].float().mean()
#         # else:
#         #     per_class_recall[c] = 1.0  # No samples for this class
#     balanced_accuracy = per_class_recall.mean()
#     sk_balanced_accuracy = sk_balanced_topk_accuracy(y_pred, y, k)
#     assert abs(balanced_accuracy.item() - sk_balanced_accuracy) < 1e-5, (
#         f"Mismatch with sklearn balanced accuracy :\n"
#         f"\tSklearn {sk_balanced_accuracy:.2f},\n"
#         f"\tImplemented {balanced_accuracy.item():.2f}"
#     )
#     return balanced_accuracy


class MetricTracker:
    """Abstract base class for metric trackers."""

    def to(self, device):
        """Move all internal tensors to the specified device."""
        return self  # By default, do nothing. Subclasses can override if they have tensors to move.

    def reset(self):
        """Reset the internal state."""
        raise NotImplementedError

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Update the internal state with new predictions and true labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits or probabilities of shape (batch_size, num_classes).
        y : torch.Tensor
            The true labels of shape (batch_size,).
        """
        raise NotImplementedError

    def __call__(self) -> float:
        """
        Compute the metric.

        Returns
        -------
        float
            The computed metric.
        """
        raise NotImplementedError

    def compute(self) -> float:
        """Alias for __call__."""
        return self()

    def extra_logs(
        self,
        logger,
        prefix: str = "",
        step: int | None = None,
        enable_logging: bool = True,
    ):
        """
        Log extra information if needed.

        Parameters
        ----------
        logger : Any
            The logger object.
        prefix : str
            Prefix for the log keys.
        step : int | None
            The step number for logging.
        enable_logging : bool
            Whether to print logs to console.
        """
        pass

    @staticmethod
    def _compute_topk_correct(
        y_pred: torch.Tensor, y: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        Compute a boolean tensor indicating if true label is in top-k predictions.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits or probabilities of shape (batch_size, num_classes).
        y : torch.Tensor
            The true labels of shape (batch_size,).
        k : int
            The 'k' in top-k accuracy.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) indicating correct predictions.
        """
        top_k_preds = y_pred.topk(k, dim=1).indices  # (batch_size, k)
        return (top_k_preds == y.unsqueeze(1)).any(dim=1)  # (batch_size,)


class TopKAccuracy(MetricTracker):
    """Tracker for standard top-k accuracy."""

    def __init__(self, k: int = 1):
        """
        Initialize the TopKAccuracy tracker.

        Parameters
        ----------
        k : int
            The 'k' in top-k accuracy.
        """
        self.k = k
        self._correct = 0
        self._total = 0

    def reset(self):
        """Reset the internal state."""
        self._correct = 0
        self._total = 0

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Update the internal state with new predictions and true labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits or probabilities of shape (batch_size, num_classes).
        y : torch.Tensor
            The true labels of shape (batch_size,).
        """
        correct = self._compute_topk_correct(y_pred, y, self.k)
        self._correct += correct.sum().item()
        self._total += y.size(0)

    def __call__(self) -> float:
        """
        Compute the top-k accuracy.

        Returns
        -------
        float
            The top-k accuracy.
        """
        if self._total == 0:
            return 0.0
        return self._correct / self._total


class BalancedTopKAccuracy(MetricTracker):
    """Tracker for balanced top-k accuracy (macro-averaged recall across classes)."""

    def __init__(self, num_classes: int, k: int = 1):
        """
        Initialize the BalancedTopKAccuracy tracker.

        Parameters
        ----------
        num_classes : int
            The number of classes.
        k : int
            The 'k' in top-k accuracy.
        """
        self.num_classes = num_classes
        self.k = k
        self._correct_per_class = torch.zeros(self.num_classes, dtype=torch.int64)
        self._count_per_class = torch.zeros(self.num_classes, dtype=torch.int64)

    def reset(self):
        """Reset the internal state."""
        self._correct_per_class = torch.zeros(self.num_classes, dtype=torch.int64)
        self._count_per_class = torch.zeros(self.num_classes, dtype=torch.int64)

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Update the internal state with new predictions and true labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits or probabilities of shape (batch_size, num_classes).
        y : torch.Tensor
            The true labels of shape (batch_size,).
        """
        correct = self._compute_topk_correct(y_pred, y, self.k)
        # Move tensors to CPU for accumulation
        y_cpu = y.detach().cpu()
        correct_cpu = correct.detach().cpu()

        for c in range(self.num_classes):
            class_mask = y_cpu == c
            self._count_per_class[c] += class_mask.sum()
            self._correct_per_class[c] += correct_cpu[class_mask].sum()

    def __call__(self) -> float:
        """
        Compute the balanced top-k accuracy.

        Returns
        -------
        float
            The balanced top-k accuracy (macro-averaged recall).

        Raises
        ------
        ValueError
            If any class has no samples.
        """
        # Check that all classes have samples
        exists_missing_classes = (self._count_per_class == 0).any().item()
        if exists_missing_classes:
            missing_classes = [
                c for c in range(self.num_classes) if self._count_per_class[c] == 0
            ]
            raise ValueError(
                f"Cannot compute balanced accuracy: classes {missing_classes} have no samples. "
                f"All {self.num_classes} classes must be represented."
            )

        # Compute per-class recall
        per_class_recall = self._correct_per_class.float() / self._count_per_class.float()

        return per_class_recall.mean().item()

    def extra_logs(
        self,
        logger,
        prefix: str = "",
        step: int | None = None,
        enable_logging: bool = True,
    ):
        """
        Log per-class recall.

        Parameters
        ----------
        logger : Any
            The logger object.
        prefix : str
            Prefix for the log keys.
        step : int | None
            The step number for logging.
        enable_logging : bool
            Whether to print logs to console.
        """
        console_logger = logging.getLogger(__name__)
        for c in range(self.num_classes):
            if self._count_per_class[c] > 0:
                class_recall = (
                    self._correct_per_class[c].float() / self._count_per_class[c].float()
                ).item()

                logger(f"{prefix}class_{c}_recall", class_recall, step=step)
                if enable_logging:
                    console_logger.info(f"{prefix}Class {c} Recall: {class_recall:.4f}")
            else:
                logger(
                    f"{prefix}class_{c}_recall",
                    float("nan"),
                    step=step,
                )
                if enable_logging:
                    console_logger.warning(f"{prefix}Class {c} Recall: N/A (no samples)")


class EGGCustom(MetricTracker):
    """Class for EGG custom metric tracking.
    We assume binary classification with labels 0 and 1.
    And y_pred are logits for class 0 and class 1. (shape (batch_size, 2))
    to compute probabilities we apply softmax on y_pred.

    The main metric returned by __call__ is the balanced accuracy.
    The metric logged with extra_logs are:
    - accuracy
    - balanced accuracy
    - f1 score
    - f2 score
    - f0.5 score
    - precision-recall Average Precision (AP, step function)
    - precision-recall AUC (trapezoidal)
    """

    def __init__(self, threshold: float = 0.5, device=None):
        """
        Initialize the EGGCustom tracker.

        Parameters
        ----------
        threshold : float
            Threshold for converting probabilities to binary predictions.
            Default is 0.5.
        device : torch.device, str, or None
            Device to place all metric objects and internal tensors on. If None, will use the device of the first input in update().
        """
        self.threshold = threshold
        self.device = torch.device(device) if device is not None else None

        # Core metrics using torchmetrics (Single Responsibility: each metric handles one thing)
        self._accuracy = BinaryAccuracy(threshold=threshold)
        self._f1 = BinaryFBetaScore(beta=1.0, threshold=threshold)
        self._f2 = BinaryFBetaScore(beta=2.0, threshold=threshold)
        self._f05 = BinaryFBetaScore(beta=0.5, threshold=threshold)
        self._pr_ap = BinaryAveragePrecision()  # Average Precision (step function)
        self._pr_curve = BinaryPrecisionRecallCurve()  # For trapezoidal AUC

        # Move metrics to device if specified
        if self.device is not None:
            self._accuracy = self._accuracy.to(self.device)
            self._f1 = self._f1.to(self.device)
            self._f2 = self._f2.to(self.device)
            self._f05 = self._f05.to(self.device)
            self._pr_ap = self._pr_ap.to(self.device)
            self._pr_curve = self._pr_curve.to(self.device)

        # For balanced accuracy, we track per-class metrics manually
        self._correct_per_class = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._count_per_class = torch.zeros(2, dtype=torch.int64, device=self.device)

    def to(self, device: torch.device):
        self.device = torch.device(device)
        self._accuracy = self._accuracy.to(self.device)
        self._f1 = self._f1.to(self.device)
        self._f2 = self._f2.to(self.device)
        self._f05 = self._f05.to(self.device)
        self._pr_ap = self._pr_ap.to(self.device)
        self._pr_curve = self._pr_curve.to(self.device)
        self._correct_per_class = self._correct_per_class.to(self.device)
        self._count_per_class = self._count_per_class.to(self.device)
        return self

    def reset(self):
        """Reset all internal states."""
        self._accuracy.reset()
        self._f1.reset()
        self._f2.reset()
        self._f05.reset()
        self._pr_ap.reset()
        self._pr_curve.reset()
        self._correct_per_class = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._count_per_class = torch.zeros(2, dtype=torch.int64, device=self.device)

    def _logits_to_probs(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities using softmax.

        Parameters
        ----------
        y_pred : torch.Tensor
            Logits of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Probability of class 1 of shape (batch_size,).
        """
        probs = torch.softmax(y_pred, dim=1)
        return probs[:, 1]  # Return probability of class 1

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Update the internal state with new predictions and true labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            Logits for class 0 and 1 of shape (batch_size, 2).
        y : torch.Tensor
            True labels of shape (batch_size,) with values 0 or 1.
        """
        # Move to correct device if needed
        if self.device is not None:
            y_pred = y_pred.to(self.device)
            y = y.to(self.device)

        # Convert logits to probability of class 1
        prob_class_1 = self._logits_to_probs(y_pred)

        # Update torchmetrics (they expect probabilities and targets)
        self._accuracy.update(prob_class_1, y)
        self._f1.update(prob_class_1, y)
        self._f2.update(prob_class_1, y)
        self._f05.update(prob_class_1, y)
        self._pr_ap.update(prob_class_1, y)
        self._pr_curve.update(prob_class_1, y)

        # Update balanced accuracy tracking
        predictions = (prob_class_1 >= self.threshold).long()
        correct = predictions == y

        # Use device-aware tensors for accumulation
        y_dev = y.detach()
        correct_dev = correct.detach()

        for c in range(2):
            class_mask = y_dev == c
            self._count_per_class[c] += class_mask.sum()
            self._correct_per_class[c] += correct_dev[class_mask].sum()

    def _compute_balanced_accuracy(self) -> torch.Tensor:
        """
        Compute balanced accuracy (macro-averaged recall).

        Returns
        -------
        torch.Tensor
            The balanced accuracy.

        Raises
        ------
        ValueError
            If any class has no samples.
        """
        if (self._count_per_class == 0).any().item():
            missing_classes = [c for c in range(2) if self._count_per_class[c] == 0]
            raise ValueError(
                f"Cannot compute balanced accuracy: classes {missing_classes} have no samples."
            )

        per_class_recall = self._correct_per_class.float() / self._count_per_class.float()
        return per_class_recall.mean()

    def __call__(self) -> torch.Tensor:
        """
        Compute the main metric (balanced accuracy).

        Returns
        -------
        torch.Tensor
            The balanced accuracy.
        """
        return self._compute_balanced_accuracy()

    def extra_logs(
        self,
        logger,
        prefix: str = "",
        step: int | None = None,
        enable_logging: bool = True,
    ):
        """
        Log extra metrics: accuracy, balanced accuracy, f1, f2, f0.5, PR-AP, PR-AUC.

        Parameters
        ----------
        logger : Any
            The logger object (callable with signature logger(key, value, step=step)).
        prefix : str
            Prefix for the log keys.
        step : int | None
            The step number for logging.
        enable_logging : bool
            Whether to print logs to console.
        """
        console_logger = logging.getLogger(__name__)

        def _safe_compute(metric_obj, metric_name: str) -> float:
            """Compute a torchmetrics value without crashing when no samples were seen."""
            try:
                return metric_obj.compute().item()
            except ValueError as e:
                if enable_logging:
                    console_logger.warning(f"{prefix}{metric_name}: N/A ({e})")
                return float("nan")

        total_samples = int(self._count_per_class.sum().item())
        if total_samples == 0:
            if enable_logging:
                console_logger.warning(
                    f"{prefix}No samples seen for metric logging. Logging NaN values."
                )

            for name in [
                "accuracy",
                "balanced_accuracy",
                "f1_score",
                "f2_score",
                "f0.5_score",
                "pr_ap",
                "pr_auc",
                "class_0_recall",
                "class_1_recall",
            ]:
                logger(f"{prefix}{name}", float("nan"), step=step)
            return

        # Compute trapezoidal PR-AUC from the curve using torchmetrics auc
        try:
            precision, recall, _ = self._pr_curve.compute()
            pr_auc_trapz = auc(recall, precision, reorder=True).item()
        except ValueError as e:
            if enable_logging:
                console_logger.warning(f"{prefix}pr_auc: N/A ({e})")
            pr_auc_trapz = float("nan")

        try:
            balanced_accuracy = self._compute_balanced_accuracy().item()
        except ValueError as e:
            if enable_logging:
                console_logger.warning(f"{prefix}balanced_accuracy: N/A ({e})")
            balanced_accuracy = float("nan")

        # Compute all metrics
        metrics = {
            "accuracy": _safe_compute(self._accuracy, "accuracy"),
            "balanced_accuracy": balanced_accuracy,
            "f1_score": _safe_compute(self._f1, "f1_score"),
            "f2_score": _safe_compute(self._f2, "f2_score"),
            "f0.5_score": _safe_compute(self._f05, "f0.5_score"),
            "pr_ap": _safe_compute(
                self._pr_ap, "pr_ap"
            ),  # Average Precision (step function)
            "pr_auc": pr_auc_trapz,  # Trapezoidal AUC
        }

        # Log all metrics
        for name, value in metrics.items():
            logger(f"{prefix}{name}", value, step=step)
            if enable_logging:
                console_logger.info(f"{prefix}{name}: {value:.4f}")

        # Log per-class recall
        for c in range(2):
            if self._count_per_class[c] > 0:
                class_recall = (
                    self._correct_per_class[c].float() / self._count_per_class[c].float()
                ).item()
                logger(f"{prefix}class_{c}_recall", class_recall, step=step)
                if enable_logging:
                    console_logger.info(f"{prefix}class_{c}_recall: {class_recall:.4f}")
            else:
                logger(f"{prefix}class_{c}_recall", float("nan"), step=step)
                if enable_logging:
                    console_logger.warning(f"{prefix}class_{c}_recall: N/A (no samples)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TopKAccuracy and BalancedTopKAccuracy")
    print("=" * 60)

    # Test 1: Basic TopKAccuracy
    print("\n--- Test 1: TopKAccuracy (k=1) ---")
    y_pred = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    y = torch.tensor([2, 0])

    topk_tracker = TopKAccuracy(k=1)
    topk_tracker.update(y_pred, y)
    print(f"Predictions: {y_pred.argmax(dim=1).tolist()}, True: {y.tolist()}")
    print(
        f"TopKAccuracy (k=1): {topk_tracker():.4f}"
    )  # Expected: 0.5 (1 correct out of 2)

    # Verify against function
    print(f"topk_accuracy function: {topk_accuracy(y_pred, y, k=1):.4f}")

    # Test 2: TopKAccuracy with k=2
    print("\n--- Test 2: TopKAccuracy (k=2) ---")
    topk_tracker_k2 = TopKAccuracy(k=2)
    topk_tracker_k2.update(y_pred, y)
    print(f"TopKAccuracy (k=2): {topk_tracker_k2():.4f}")  # Expected: 1.0 (both in top-2)

    # Test 3: BalancedTopKAccuracy with imbalanced data (should raise error)
    print("\n--- Test 3: BalancedTopKAccuracy (missing class raises error) ---")
    # 4 samples of class 0, all predicted as class 1
    y_imbalanced = torch.tensor([0, 0, 0, 0])
    y_pred_imbalanced = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])

    balanced_tracker = BalancedTopKAccuracy(num_classes=2, k=1)
    balanced_tracker.update(y_pred_imbalanced, y_imbalanced)
    try:
        result = balanced_tracker()
        print(f"ERROR: Should have raised ValueError but got {result:.4f}")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    # Test 4: BalancedTopKAccuracy with all classes present
    print("\n--- Test 4: BalancedTopKAccuracy (all classes present) ---")
    y_balanced = torch.tensor([0, 0, 1, 1])
    y_pred_balanced = torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]
    )  # 1 correct per class (50% each)

    balanced_tracker3 = BalancedTopKAccuracy(num_classes=2, k=1)
    balanced_tracker3.update(y_pred_balanced, y_balanced)
    result3 = balanced_tracker3()
    print("1 correct out of 2 for each class")
    print(f"BalancedTopKAccuracy: {result3:.4f}")  # Expected: 0.5

    # Test 5: Multiple batches accumulation
    print("\n--- Test 5: Multiple batches accumulation ---")
    topk_multi = TopKAccuracy(k=1)
    balanced_multi = BalancedTopKAccuracy(num_classes=3, k=1)

    # Batch 1
    y1 = torch.tensor([0, 1, 2])
    y_pred1 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )  # All correct
    topk_multi.update(y_pred1, y1)
    balanced_multi.update(y_pred1, y1)

    # Batch 2
    y2 = torch.tensor([0, 1, 2])
    y_pred2 = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )  # All wrong
    topk_multi.update(y_pred2, y2)
    balanced_multi.update(y_pred2, y2)

    print("Batch 1: All correct, Batch 2: All wrong")
    print(f"TopKAccuracy after 2 batches: {topk_multi():.4f}")  # Expected: 0.5
    print(
        f"BalancedTopKAccuracy after 2 batches: {balanced_multi():.4f}"
    )  # Expected: 0.5

    # Test 6: Reset functionality
    print("\n--- Test 6: Reset functionality ---")
    topk_multi.reset()
    balanced_multi.reset()
    print(f"After reset - TopKAccuracy: {topk_multi():.4f}")  # Expected: 0.0
    try:
        balanced_multi()
        print("ERROR: Should have raised ValueError after reset")
    except ValueError:
        print(
            "After reset - BalancedTopKAccuracy correctly raises ValueError (no samples)"
        )

    print("\n" + "=" * 60)
    print("Testing EGGCustom")
    print("=" * 60)

    # Test 7: EGGCustom basic functionality
    print("\n--- Test 7: EGGCustom basic (perfect predictions) ---")
    egg_tracker = EGGCustom()
    # Logits: high for correct class
    y_pred_egg = torch.tensor([[-2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, -2.0]])
    y_egg = torch.tensor([1, 1, 0, 0])  # All predictions should be correct
    egg_tracker.update(y_pred_egg, y_egg)

    balanced_acc = egg_tracker()
    print(f"Balanced Accuracy (perfect): {balanced_acc:.4f}")  # Expected: 1.0
    assert abs(balanced_acc - 1.0) < 1e-5, f"Expected 1.0, got {balanced_acc}"
    print("✓ Perfect predictions test passed")

    # Test 8: EGGCustom with 50% accuracy per class
    print("\n--- Test 8: EGGCustom (50% per class) ---")
    egg_tracker2 = EGGCustom()
    # 2 class 0, 2 class 1; 1 correct per class
    y_pred_egg2 = torch.tensor(
        [
            [2.0, -2.0],  # predicts 0, true 0 -> correct
            [-2.0, 2.0],  # predicts 1, true 0 -> wrong
            [2.0, -2.0],  # predicts 0, true 1 -> wrong
            [-2.0, 2.0],  # predicts 1, true 1 -> correct
        ]
    )
    y_egg2 = torch.tensor([0, 0, 1, 1])
    egg_tracker2.update(y_pred_egg2, y_egg2)

    balanced_acc2 = egg_tracker2()
    print(f"Balanced Accuracy (50% each): {balanced_acc2:.4f}")  # Expected: 0.5
    assert abs(balanced_acc2 - 0.5) < 1e-5, f"Expected 0.5, got {balanced_acc2}"
    print("✓ 50% accuracy test passed")

    # Test 9: EGGCustom extra_logs
    print("\n--- Test 9: EGGCustom extra_logs ---")
    logged_metrics = {}

    def mock_logger(key, value, step=None):
        logged_metrics[key] = value

    egg_tracker2.extra_logs(mock_logger, prefix="test_", enable_logging=True)

    expected_keys = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_score",
        "test_f2_score",
        "test_f0.5_score",
        "test_pr_ap",
        "test_pr_auc",
        "test_class_0_recall",
        "test_class_1_recall",
    ]
    for key in expected_keys:
        assert key in logged_metrics, f"Missing key: {key}"
        print(f"  {key}: {logged_metrics[key]:.4f}")
    print("✓ Extra logs test passed")

    # Test 10: EGGCustom reset
    print("\n--- Test 10: EGGCustom reset ---")
    egg_tracker2.reset()
    try:
        egg_tracker2()
        print("ERROR: Should have raised ValueError after reset")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError after reset: {e}")

    # Test 11: EGGCustom with multiple batches
    print("\n--- Test 11: EGGCustom multiple batches ---")
    egg_multi = EGGCustom()

    # Batch 1: all correct
    y_pred_b1 = torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
    y_b1 = torch.tensor([1, 0])
    egg_multi.update(y_pred_b1, y_b1)

    # Batch 2: all wrong
    y_pred_b2 = torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
    y_b2 = torch.tensor([0, 1])
    egg_multi.update(y_pred_b2, y_b2)

    balanced_acc_multi = egg_multi()
    print(f"Balanced Accuracy (2 batches, 50%): {balanced_acc_multi:.4f}")
    assert abs(balanced_acc_multi - 0.5) < 1e-5, f"Expected 0.5, got {balanced_acc_multi}"
    print("✓ Multiple batches test passed")

    # Test 12: EGGCustom with imbalanced data
    print("\n--- Test 12: EGGCustom imbalanced data ---")
    egg_imb = EGGCustom()
    # 4 samples class 0, 2 samples class 1
    # class 0: 3/4 correct, class 1: 1/2 correct
    y_pred_imb = torch.tensor(
        [
            [2.0, -2.0],  # pred 0, true 0 -> correct
            [2.0, -2.0],  # pred 0, true 0 -> correct
            [2.0, -2.0],  # pred 0, true 0 -> correct
            [-2.0, 2.0],  # pred 1, true 0 -> wrong
            [-2.0, 2.0],  # pred 1, true 1 -> correct
            [2.0, -2.0],  # pred 0, true 1 -> wrong
        ]
    )
    y_imb = torch.tensor([0, 0, 0, 0, 1, 1])
    egg_imb.update(y_pred_imb, y_imb)

    balanced_acc_imb = egg_imb()
    # class 0 recall: 3/4 = 0.75, class 1 recall: 1/2 = 0.5
    # balanced acc = (0.75 + 0.5) / 2 = 0.625
    print(f"Balanced Accuracy (imbalanced): {balanced_acc_imb:.4f}")
    expected_imb = (0.75 + 0.5) / 2
    assert (
        abs(balanced_acc_imb - expected_imb) < 1e-5
    ), f"Expected {expected_imb}, got {balanced_acc_imb}"
    print("✓ Imbalanced data test passed")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
