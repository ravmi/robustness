from clearml import Logger

class CLearMlLogger():
    def __init__(self, plot_name):
        self.log = Logger.current_logger()
        self.plot_name = plot_name
        self.iteration = 0

    def log_scalar(self, name, scalar):
        self.log.log_scalar(self.plot_name, name, self.iteration, scalar)
        self.iteration += 1