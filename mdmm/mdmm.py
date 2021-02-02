"""The Modified Differential Multiplier Method (MDMM) for PyTorch."""

import abc

import torch
from torch import nn, optim


class Constraint(nn.Module, metaclass=abc.ABCMeta):
    """The base class for all constraint types."""

    def __init__(self, fn, scale, damping):
        super().__init__()
        self.fn = fn
        self.register_buffer('scale', torch.as_tensor(scale))
        self.register_buffer('damping', torch.as_tensor(damping))
        self.lmbda = nn.Parameter(torch.tensor(0.))

    @abc.abstractmethod
    def infeasibility(self, fn_value):
        ...

    def forward(self):
        fn_value = self.fn()
        inf = self.infeasibility(fn_value)
        l_term = self.lmbda * inf
        damp_term = self.damping * inf**2 / 2
        return self.scale * (damp_term - l_term), fn_value


class EqConstraint(Constraint):
    """Represents an equality constraint."""

    def __init__(self, fn, value, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('value', torch.as_tensor(value))

    def extra_repr(self):
        return f'value={self.value:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value):
        return self.value - fn_value


class MaxConstraint(Constraint):
    """Represents a maximum inequality constraint which uses a slack variable."""

    def __init__(self, fn, max, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        fn_value = self.fn()
        self.register_buffer('max', fn_value.new_tensor(max))
        self.slack = nn.Parameter((self.max - fn_value).relu().pow(1/2))

    def extra_repr(self):
        return f'max={self.max:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value):
        return self.max - fn_value - self.slack**2


class MaxConstraintHard(Constraint):
    """Represents a maximum inequality constraint without a slack variable."""

    def __init__(self, fn, max, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'max={self.max:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value):
        return fn_value.clamp(max=self.max) - fn_value


class MinConstraint(Constraint):
    """Represents a minimum inequality constraint which uses a slack variable."""

    def __init__(self, fn, min, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        fn_value = self.fn()
        self.register_buffer('min', fn_value.new_tensor(min))
        self.slack = nn.Parameter((fn_value - self.min).relu().pow(1/2))

    def extra_repr(self):
        return f'min={self.min:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value):
        return fn_value - self.min - self.slack**2


class MinConstraintHard(Constraint):
    """Represents a minimum inequality constraint without a slack variable."""

    def __init__(self, fn, min, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('min', torch.as_tensor(min))

    def extra_repr(self):
        return f'min={self.min:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value):
        return fn_value.clamp(min=self.min) - fn_value


class BoundConstraintHard(Constraint):
    """Represents a bound constraint."""

    def __init__(self, fn, min, max, scale=1., damping=1.):
        super().__init__(fn, scale, damping)
        self.register_buffer('min', torch.as_tensor(min))
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'min={self.min:g}, max={self.max:g}, ' \
               f'scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value):
        return fn_value.clamp(self.min, self.max) - fn_value


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
        fn_values = []
        for c in self:
            penalty, fn_value = c()
            output += penalty
            fn_values.append(fn_value)
        return output, fn_values
