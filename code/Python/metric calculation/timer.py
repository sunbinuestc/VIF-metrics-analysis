import time


class Timer:
    def __init__(self, name=None, cache=None):
        self.name = name
        self.cache = cache
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_value, traceback):
        time_cost = time.time()-self.start
        if self.name:
            print(f"{self.name}", f"time_cost: {time_cost}")
        else:
            print(f"time_cost: {time_cost}")
        if self.cache is not None:
            self.cache.append(time_cost)
