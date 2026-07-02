import time
from contextlib import contextmanager

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        
    def stop(self):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration

@contextmanager
def measure_time():
    timer = Timer()
    timer.start()
    yield timer
    timer.stop()
