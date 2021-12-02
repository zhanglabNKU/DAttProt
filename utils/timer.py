import time


class Timer:
    def __init__(self, start_now=False):
        self.t_start = time.time() if start_now else None
        self.time_count = 0.

    def start(self):
        self.t_start = time.time()

    def stop(self):
        period = time.time() - self.t_start
        self.time_count += period
        self.t_start = None
        return period

    def reset(self):
        self.__init__()

    def restart(self):
        self.__init__(True)

    def __repr__(self):
        t_now = int(self.time_count + time.time() - self.t_start)
        h, m, s = t_now // 3600, t_now % 3600 // 60, t_now % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
