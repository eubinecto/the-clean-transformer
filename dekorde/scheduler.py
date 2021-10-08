import torch.optim as optm
from torch.optim import Optimizer
import warnings


class TransformerScheduler(optm.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, embed_dim: int) -> None:
        self.warmup_steps = warmup_steps
        self.embed_dim = embed_dim
        super().__init__(optimizer)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [ self._last_lr['lr'] for group in self.optimizer.param_groups]

    def calc_lr(self):
        return self.embed_dim **(-0.5) * min(self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5))

    def step(self):
        self._step_count += 1

        values = self.calc_lr()
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

