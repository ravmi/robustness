import numpy as np

#iou
#pixel accuracy
# f1


class Metric():
    def class_imbalance(self, predicted, truth):
        argp = np.argmax(predicted, axis=0).reshape(-1)
        argt = np.argmax(truth, axis=0).reshape(-1)

        return np.all(argp == 0) or np.all(argp == 1)

    def pixel_accuracy(self, predicted, truth):
        argp = np.argmax(predicted, axis=0).reshape(-1)
        argt = np.argmax(truth, axis=0).reshape(-1)

        return np.mean(argp == argt)

    def recall(self, predicted, truth):
        argp = np.argmax(predicted, axis=0).reshape(-1)
        argt = np.argmax(truth, axis=0).reshape(-1)

        positive = argp == 1
        negative = argp == 0

        true_positive = positive & (argp == argt)
        false_negative = negative & (argp != argt)

        tpi = np.sum(true_positive)
        fni = np.sum(false_negative)

        if tpi + fni == 0:
            return None
        return tpi / (tpi + fni)

    def precision(self, predicted, truth):
        argp = np.argmax(predicted, axis=0).reshape(-1)
        argt = np.argmax(truth, axis=0).reshape(-1)

        positive = argp == 1
        true_positive = positive & (argp == argt)
        false_positive = positive & (argp != argt)

        tpi = np.sum(true_positive)
        fpi = np.sum(false_positive)

        if tpi + fpi == 0:
            return None
        return tpi / (tpi + fpi)

    def balanced(self, predicted, truth):
        prec = self.precision(predicted, truth)
        rec = self.recall(predicted, truth)

        if prec == None or rec == None:
            return None

        return (prec + rec) / 2

    def top5(self, predicted, truth, radius=15, samples=5):

        y_limit, x_limit = truth.shape[1:3]
        pcopy = predicted[1].copy()
        correct = 0
        for i in range(5):
            y, x = np.unravel_index(pcopy.argmax(), pcopy.shape)

            y_min = max(0, y - radius)
            y_max = min(y_limit, y + radius)

            x_min = max(0, x - radius)
            x_max = min(x_limit, y + radius)

            pcopy[y_min:y_max+1, x_min:x_max+1] = -np.inf

            correct += int(truth[1][y][x] == 1)
        return correct / 5


    def __init__(self, metric):
        """Assumes that input arrays have two channels,
        where the first denotes NO and the second denotes YES.
        The channel can a first or second dim, depending if we use
        batch or sample (both are supported)."""

        self.metric_name = metric
        if metric == "pixel_accuracy":
            self.metric = self.pixel_accuracy
        elif metric == "recall":
            self.metric = self.recall
        elif metric == "precision":
            self.metric = self.precision
        elif metric == "balanced":
            self.metric = self.balanced
        elif metric == "top5":
            self.metric = self.top5
        elif metric == "class_imbalance":
            self.metric = self.class_imbalance
        else:
            raise ValueError("Uncorrect metric given")

        self.measurements = list()

    def measure(self, predicted, truth):
        assert predicted.shape == truth.shape

        if predicted.ndim == 4:
            for i in range(len(predicted)):
                measurement = self.metric(predicted[i], truth[i])
                if measurement is not None:
                    self.measurements.append(measurement)
        else:
            assert predicted.ndim == 3
            measurement = self.metric(predicted, truth)
            if measurement is not None:
                self.measurements.append(measurement)

    def total(self):
        if len(self.measurements) == 0:
            return 0.
        return sum(self.measurements) / len(self.measurements)
