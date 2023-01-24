from timeit import default_timer as timer
import torch
import logging
import inspect


class time_measurer():
    def __init__(self, units='ms', desc=None):
        self.start_time = timer()
        self.units = units
        self.desc = desc

    def __call__(self):
        return self.elapsed()

    def elapsed(self):
        value = float(timer() - self.start_time)
        if self.units == 'ms':
            value = float(f'{(1000 * value):.1f}')
        return value

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        caller_name = inspect.getmodule(inspect.stack()[1][0]).__name__
        logger = logging.getLogger(caller_name)
        ms_elapsed = self.elapsed()
        logger.debug(f"{self.desc}: {ms_elapsed}ms")


class cuda_time_measurer():
    """ https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    https://auro-227.medium.com/timing-your-pytorch-code-fragments-e1a556e81f2 """

    def __init__(self, units=None):
        # self.start_time = timer()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.units = units

        self.start_event.record()

    def __call__(self):
        self.end_event.record()
        torch.cuda.synchronize()
        value = self.start_event.elapsed_time(self.end_event)
        assert self.units == 'ms'
        return value
