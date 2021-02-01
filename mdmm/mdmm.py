"""The Modified Differential Multiplier Method (MDMM)."""

import abc

import torch
from torch import nn, optim


class Constraint(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, fn, damping):
        super().__init__()
        self.fn = fn
        self.register_buffer('damping', torch.as_tensor(damping))
        self.lmbda = nn.Parameter(torch.tensor(0.))

    @abc.abstractmethod
    def c_value(self, loss):
        ...

    def forward(self):
        loss = self.fn()
        c_value = self.c_value(loss)
        output = self.damping * c_value**2 / 2 - self.lmbda * c_value
        return output, loss


class EqConstraint(Constraint):
    def __init__(self, fn, value, damping=1e-2):
        super().__init__(fn, damping)
        self.register_buffer('value', torch.as_tensor(value))

    def extra_repr(self):
        return f'value={self.value:g}, damping={self.damping:g}'

    def c_value(self, loss):
        return self.value - loss


class MaxConstraint(Constraint):
    def __init__(self, fn, max, damping=1e-2):
        super().__init__(fn, damping)
        loss = self.fn()
        self.register_buffer('max', loss.new_tensor(max))
        self.slack = nn.Parameter((self.max - loss).relu().pow(1/2))

    def extra_repr(self):
        return f'max={self.max:g}, damping={self.damping:g}'

    def c_value(self, loss):
        return self.max - loss - self.slack**2


class MaxConstraintHard(Constraint):
    def __init__(self, fn, max, damping=1e-2):
        super().__init__(fn, damping)
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'max={self.max:g}, damping={self.damping:g}'

    def c_value(self, loss):
        return loss.clamp(max=self.max) - loss


class MinConstraint(Constraint):
    def __init__(self, fn, min, damping=1e-2):
        super().__init__(fn, damping)
        loss = self.fn()
        self.register_buffer('min', loss.new_tensor(min))
        self.slack = nn.Parameter((loss - self.min).relu().pow(1/2))

    def extra_repr(self):
        return f'min={self.min:g}, damping={self.damping:g}'

    def c_value(self, loss):
        return loss - self.min - self.slack**2


class MinConstraintHard(Constraint):
    def __init__(self, fn, min, damping=1e-2):
        super().__init__(fn, damping)
        self.register_buffer('min', torch.as_tensor(min))

    def extra_repr(self):
        return f'min={self.min:g}, damping={self.damping:g}'

    def c_value(self, loss):
        return loss.clamp(min=self.min) - loss


class BoundConstraintHard(Constraint):
    def __init__(self, fn, min, max, damping=1e-2):
        super().__init__(fn, damping)
        self.register_buffer('min', torch.as_tensor(min))
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'min={self.min:g}, max={self.max:g}, damping={self.damping:g}'

    def c_value(self, loss):
        return loss.clamp(self.min, self.max) - loss


class MDMM(nn.ModuleList):
    def make_optimizer(self, params, *, optimizer=optim.Adamax, lr=2e-3):
        lambdas = [c.lmbda for c in self]
        slacks = [c.slack for c in self if hasattr(c, 'slack')]
        return optimizer([{'params': params, 'lr': lr},
                          {'params': lambdas, 'lr': -lr},
                          {'params': slacks, 'lr': lr}])

    def forward(self, loss):
        output = loss.clone()
        losses = []
        for c in self:
            c_value, c_loss = c()
            output += c_value
            losses.append(c_loss)
        return output, losses
