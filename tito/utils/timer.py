import cProfile
import io
import pstats
import re
import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.timings = {}

    @contextmanager
    def time(self, label):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            self.timings[label] = (end - start) * 1000

    def get(self, label):
        return self.timings[label]

    def get_timings(self):
        return self.timings


class Profiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stream = io.StringIO()

    @contextmanager
    def time(self):
        try:
            self.profiler.enable()
            yield
        finally:
            self.profiler.disable()
            self.print()

    def filter_output(self, output):
        return output

    def get_output(self):
        ps = pstats.Stats(self.profiler, stream=self.stream).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        raw_output = self.stream.getvalue()
        filtered_output = self.filter_output(raw_output)
        return filtered_output

    def print(self):
        print(self.get_output())
