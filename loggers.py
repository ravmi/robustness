from clearml import Logger
from collections import defaultdict

class ClearMLLogger():
    def __init__(self, plot_name):
        self.log = Logger.current_logger()
        self.plot_name = plot_name
        self.iteration = defaultdict(lambda: 0)

    def report_scalar(self, plot_name, series_name, scalar):
        key = (plot_name, series_name)
        self.log.report_scalar(plot_name, series_name, iteration=self.iteration[key], value=scalar)
        self.iteration[key] += 1
