# Nguyễn Tâm Bình - 2324801030003
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve


def best_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def binary_metrics(labels, scores, threshold=None, predictions=None):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if predictions is None:
        predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    both_classes = len(np.unique(labels)) == 2
    return {"threshold": threshold, "accuracy": accuracy_score(labels, predictions), "precision": precision, "recall": recall, "f1": f1, "auroc": roc_auc_score(labels, scores) if both_classes else float("nan"), "auprc": average_precision_score(labels, scores) if both_classes else float("nan")}


def fixed_metrics(labels, scores, threshold=0.0):

    metrics = binary_metrics(labels, scores, threshold)
    return {f"{name}_05": metrics[name] for name in ("accuracy", "precision", "recall", "f1")}


def best_accuracy_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    positives = np.sum(np.asarray(labels) == 1)
    accuracy = (tpr * positives + (1 - fpr) * (len(labels) - positives)) / len(labels)
    return float(thresholds[int(np.argmax(accuracy[1:])) + 1])


def accuracy_metrics(valid_labels, valid_scores, labels, scores):

    threshold = best_accuracy_threshold(valid_labels, valid_scores)
    metrics = binary_metrics(labels, scores, threshold)
    return {"threshold_acc": threshold, **{f"{name}_acc": metrics[name] for name in ("accuracy", "precision", "recall", "f1")}}






