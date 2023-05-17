import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=config["optimizer"]["betas"],
            eps=config["optimizer"]["eps"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = config["optimizer"]["warm_up_step"]
        self.anneal_steps = config["optimizer"]["anneal_steps"]
        self.anneal_rate = config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(config["model"]["dim"]*4, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
