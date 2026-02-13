from typing import Protocol


class Logger(Protocol):

    def log_metrics(self, step: int, metrics: dict[str, float]): ...

    def close(self): ...
