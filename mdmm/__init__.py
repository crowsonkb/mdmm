"""The Modified Differential Multiplier Method (MDMM) for PyTorch."""

from .mdmm import (ConstraintReturn, Constraint, EqConstraint, MaxConstraint, MaxConstraintHard,
                   MinConstraint, MinConstraintHard, BoundConstraintHard, MDMMReturn, MDMM)

__version__ = '0.1.3'
