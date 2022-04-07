import numpy as np
from dataclasses import dataclass


def confusion_matrix(predicted, truth):
    """
    Returns a dict that represents the confusion matrix.
    """
    assert predicted.ndim == 3
    assert truth.ndim == 3
    predicted = to_binary(predicted) == 1
    truth = truth == 1

    tp = np.sum(predicted & truth)
    fp = np.sum(predicted & ~truth)
    fn = np.sum(~predicted & truth)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn
    }


def to_binary(predicted: np.ndarray):
    """
    Gives binary answers of shape predicted.shape
    predicted: logits of shape (1, height, widths) - single channel

    As the input are logits, transforming it to True/False is simple comparison to 0.
    """
    assert predicted.ndim == 3

    return (predicted > 0.0) * 1.0


class AccuracyMetric():
    def __init__(self):
        self.measurements = list()

    def __str__(self):
        return self.__class__.__name__

    def measure(self, x, y):
        raise NotImplementedError

    def get_metric(self):
        measurements = [m for m in self.measurements if m is not None]
        if len(measurements) == 0:
            return 0.
        return np.mean(np.asarray(measurements))


class PixelAccuracyMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        predicted = to_binary(predicted)
        compare = (predicted == truth).reshape(-1)

        self.measurements.append(np.mean(compare))


class ClassImbalanceMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        # features, height, width
        assert predicted.ndim == 3
        guess = to_binary(predicted)
        imbalance = np.all(guess == 0) or np.all(guess == 1)

        self.measurements.append(imbalance * 1.0)


class RecallTotalMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        cm = confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fn = cm["false_negative"]

        self.measurements.append((tp, fn))

    def get_metric(self):
        total_tp = sum([tp for tp, fn in self.measurements])
        total_fn = sum([fn for tp, fn in self.measurements])

        return total_tp / (total_tp + total_fn)


class PrecisionTotalMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        cm = confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fp = cm["false_positive"]

        self.measurements.append((tp, fp))

    def get_metric(self):
        total_tp = sum([tp for tp, fp in self.measurements])
        total_fp = sum([fp for tp, fp in self.measurements])

        return total_tp / (total_tp + total_fp)


class RecallMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        cm = confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fn = cm["false_negative"]

        if fn + tp == 0:
            return None

        self.measurements.append(tp / (fn + tp))


class PrecisionMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        cm = confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fp = cm["false_positive"]

        if fp + tp == 0:
            return None

        self.measurements.append(tp / (tp + fp))


class BalancedMetric(AccuracyMetric):
    def measure(self, predicted, truth):
        cm = confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fn = cm["false_negative"]
        fp = cm["false_positive"]

        if tp + fp == 0 or fn + tp == 0:
            self.measurements.append(None)

        pc = tp / (tp + fp)
        rc = tp / (tp + fn)

        self.measurements.append((pc + rc) / 2.0)


class Top5Metric(AccuracyMetric):
    def measure(self, predicted, truth, radius=15, samples=5):
        _, y_limit, x_limit = truth.shape
        pcopy = predicted[0].copy()
        truth = truth[0]
        correct = 0

        def locate_max(p):
            return np.unravel_index(np.argmax(p), p.shape)

        for i in range(samples):
            y, x = locate_max(pcopy)

            y_min = max(0, y - radius)
            y_max = min(y_limit, y + radius)

            x_min = max(0, x - radius)
            x_max = min(x_limit, x + radius)

            pcopy[y_min:y_max+1, x_min:x_max+1] = -np.inf

            correct += int(truth[y][x] == 1)

        self.measurements.append(correct / samples)
