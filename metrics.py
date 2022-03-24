import numpy as np


class Accuracy():
    @classmethod
    def mean_accuracy(predicted, truth):
        argp = np.argmax(predicted, axis=1).reshape(-1)
        argt = np.argmax(truth, axis=1).reshape(-1)

        return np.mean(argp == argt)

    @classmethod
    def recall(predicted, truth):
        argp = np.argmax(predicted, axis=1).reshape(-1)
        argt = np.argmax(truth, axis=1).reshape(-1)

        positive = argp == 1
        negative = argp == 0

        true_positive = positive & (argp == argt)
        false_negative = negative & (argp == argt)

        tpi = np.sum(true_positive)
        fni = np.sum(false_negative)

        return tpi / (tpi + fni)

    @classmethod
    def precision(predicted, truth):
        argp = np.argmax(predicted, axis=1).reshape(-1)
        argt = np.argmax(truth, axis=1).reshape(-1)

        positive = argp == 1
        true_positive = positive & (argp == argt)
        false_positive = positive & (argp != argt)

        tpi = np.sum(true_positive)
        fpi = np.sum(false_positive)

        return tpi / (tpi + fpi)

    @classmethod
    def balanced(predicted, truth):
        prec = Accuracy.precision(predicted, truth)
        rec = Accuracy.recall(predicted, truth)

        return (prec + rec) / 2

    @classmethod
    def top5(predicted, truth, radius=15, samples=5):
        y_limit, x_limit = truth.shape[:2]
        pcopy = predicted.copy()
        for i in range(5):
            y, x = np.unravel_index(pcopy.argmax(), pcopy.shape)

            y_min = max(0, y - radius)
            y_max = min(y_limit, y + radius)

            x_min = max(0, x - radius)
            x_max = min(x_limit, y + radius)

            pcopy[:, y_min:y_max+1, x_min:x_max+1] = -np.inf


    def __init__(self, metric):
        """Assumes that input arrays have two channels,
        where the first denotes NO and the second denotes YES.
        The channel can a first or second dim, depending if we use
        batch or sample (both are supported)."""

        if metric == "mean":
            self.accuracy = mean_accuracy
        elif metric == "recall":
            self.accuracy = recall
        elif metric == "precision":
            self.accuracy = precision
        elif metric == "balanced":
            self.accuracy = balanced
        elif metric == "top5":
            self.accuracy = top5

        self.measurements = list()

        def measure(self, predicted, truth):
            assert predicted.shape == truth.shape

            if predicted.ndim == 4:
                for i in range(len(predicted)):
                    measurement = self.metric(predicted[i], truth[i])
                    self.measurements.append(measurement)
            else:
                assert predicted.ndim == 3
                measurement = self.metric(predicted, truth)
                self.measurements.append(measurement)

        def accuracy():
            return sum(self.measurements) / len(self.measurements)
