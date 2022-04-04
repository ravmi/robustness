import numpy as np

class Metric():
    def __init__(self, metric):
        """
        Assumes that the arrays have just single channel (binary classification).
        Inputs can have ndim = 3 or ndim = 4.
        """

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

    def to_binary(self, predicted: np.ndarray):
        """
        Gives binary answers of shape predicted.shape
        predicted: logits of shape (1, height, widths) - single channel

        As the input are logits, transforming it to True/False is simple comparison to 0.
        """
        assert predicted.ndim == 3

        return (predicted > 0.0) * 1.0


    def class_imbalance(self, predicted, truth):
        # features, height, width
        assert predicted.ndim == 3
        guess = self.get_binary(predicted)
        imbalance = np.all(guess == 0) or np.all(guess == 1)

        return imbalance * 1.0

    def pixel_accuracy(self, predicted, truth):
        predicted = self.to_binary(predicted)
        compare = (predicted == truth).reshape(-1)
        return np.mean(compare)


    def confusion_matrix(self, predicted, truth):
        """
        Returns a dict that represents the confusion matrix.
        """
        assert predicted.ndim == 3
        assert truth.ndim == 3
        predicted = self.to_binary(predicted)

        tp = np.sum(predicted & truth)
        fp = np.sum(predicted & ~truth)
        fn = np.sum(~predicted & truth)
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn
        }

    def recall(self, predicted, truth):
        cm = self.confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fn = cm["false_negative"]

        if fn + tp == 0:
            return None

        return tp / (fn + tp)

    def precision(self, predicted, truth):
        cm = self.confusion_matrix(predicted, truth)
        tp = cm["true_positive"]
        fp = cm["false_positive"]

        if fp + tp == 0:
            return None

        return tp / (tp + fp)

    def balanced(self, predicted, truth):
        pc = self.precision(predicted, truth)
        rc = self.recall(predicted, truth)

        if pc is None or rc is None:
            return None

        return (pc + rc) / 2.0

    def top5(self, predicted, truth, radius=15, samples=5):
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

        return correct / samples

    def measure(self, predicted, truth):
        assert predicted.shape == truth.shape
        if predicted.ndim == 3:
            predicted = np.expand_dims(predicted, axis=0)
        assert predicted.ndim == 4
        assert predicted.shape[1] == 1

        for p, t in zip(predicted, truth):
            self.measurements.append(self.metric(p, t))

    def total(self):
        measurements = [m for m in self.measurements if m is not None]
        if len(measurements) == 0:
            return None
        return np.mean(np.asarray(measurements))
