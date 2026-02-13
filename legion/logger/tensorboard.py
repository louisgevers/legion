import tensorboardX


class TensorboardLogger:
    def __init__(self, log_dir: str):
        self._writer = tensorboardX.SummaryWriter(log_dir)

    def log_metrics(self, step: int, metrics: dict[str, float]):
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def close(self):
        self._writer.close()
