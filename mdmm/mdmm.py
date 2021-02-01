"""The Modified Differential Multiplier Method (MDMM) for PyTorch."""

import abc

import torch
from torch import nn, optim


class Constraint(nn.Module, metaclass=abc.ABCMeta):
    """The parent class for all constraints."""

    def __init__(self, fn, scale, damping):
        super().__init__()
        self.fn = fn
        self.register_buffer('scale', torch.as_tensor(scale))
        self.register_buffer('damping', torch.as_tensor(damping))
        self.lmbda = nn.Parameter(torch.tensor(0.))

    @abc.abstractmethod
    def infeasibility(self, loss):
        ...

    def forward(self):
        loss = self.fn()
        inf = self.infeasibility(loss)
        l_term = self.lmbda * inf
        damp_term = self.damping * inf**2 / 2
        return self.scale * (damp_term - l_term), loss


class EqConstraint(Constraint):
    """Represents an equality constraint."""

    def __init__(self, fn, value, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('value', torch.as_tensor(value))

    def extra_repr(self):
        return f'value={self.value:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, loss):
        return self.value - loss


class MaxConstraint(Constraint):
    """Represents a maximum inequality constraint which uses a slack variable."""

    def __init__(self, fn, max, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        loss = self.fn()
        self.register_buffer('max', loss.new_tensor(max))
        self.slack = nn.Parameter((self.max - loss).relu().pow(1/2))

    def extra_repr(self):
        return f'max={self.max:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, loss):
        return self.max - loss - self.slack**2


class MaxConstraintHard(Constraint):
    """Represents a maximum inequality constraint without a slack variable."""

    def __init__(self, fn, max, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'max={self.max:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, loss):
        return loss.clamp(max=self.max) - loss


class MinConstraint(Constraint):
    """Represents a minimum inequality constraint which uses a slack variable."""

    def __init__(self, fn, min, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        loss = self.fn()
        self.register_buffer('min', loss.new_tensor(min))
        self.slack = nn.Parameter((loss - self.min).relu().pow(1/2))

    def extra_repr(self):
        return f'min={self.min:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, loss):
        return loss - self.min - self.slack**2


class MinConstraintHard(Constraint):
    """Represents a minimum inequality constraint without a slack variable."""

    def __init__(self, fn, min, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('min', torch.as_tensor(min))

    def extra_repr(self):
        return f'min={self.min:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, loss):
        return loss.clamp(min=self.min) - loss


class BoundConstraintHard(Constraint):
    """Represents a bound constraint."""

    def __init__(self, fn, min, max, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('min', torch.as_tensor(min))
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'min={self.min:g}, max={self.max:g}, ' \
               f'scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, loss):
        return loss.clamp(self.min, self.max) - loss


class MDMM(nn.ModuleList):
    """The main MDMM class, which combines multiple constraints."""

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
            penalty, c_loss = c()
            output += penalty
            losses.append(c_loss)
        return output, losses
