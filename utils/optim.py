import torch.nn as nn


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, lr, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        # lr: max learning rate in the schedule
        self.factor = lr / (model_size ** (-0.5) * warmup ** (-0.5))
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        self.set_rate(rate)
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def set_rate(self, rate):
        for p in self.optimizer.param_groups:
            p['lr'] = rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state = {
            'optim': self.optimizer.state_dict(),
            'step': self._step,
            'lr': self._rate,
            'factor': self.factor,
            'dim': self.model_size
        }
        return state

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optim'])
        self._step = state['step']
        self._rate = state['lr']
        self.factor = state['factor']
        self.model_size = state['dim']


def rate(opt):
    if isinstance(opt, NoamOpt):
        return opt.rate()
    else:
        return opt.state_dict()['param_groups'][0]['lr']


def set_rate(opt, rate):
    if isinstance(opt, NoamOpt):
        opt.set_rate(rate)
    else:
        for p in opt.param_groups:
            p['lr'] = rate


def freeze(modules, op=True):
    r"""
    >>> import torch.optim as optim
    >>> modules = [net.embedding, net.pos_embedding, net.map, net.coder]
    Use freeze(op=True) to freeze module parameters.
    >>> freeze(modules)
    >>> optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
    Use freeze(op=False) and optimizer.add_param_group({'params': module.parameters()}) to unfreeze any module.
    >>> freeze(modules, False)
    >>> for mod in modules:
    >>>     optimizer.add_param_group({'params': mod.parameters()})
    """
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for module in modules:
        if isinstance(module, nn.Parameter):
            module.requires_grad = not op
        else:
            for para in module.parameters():
                para.requires_grad = not op
