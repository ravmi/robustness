from clearml import Logger

class ClearMLLogger():
    def __init__(self, plot_name):
        self.log = Logger.current_logger()
        self.plot_name = plot_name
        self.iteration = 0
        # TODO fix this

    def report_scalar(self, plot_name, series_name, scalar):
        self.log.report_scalar(plot_name, series_name, iteration=self.iteration, value=scalar)
        self.iteration += 1
