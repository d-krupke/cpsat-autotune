"""
This module contains some utility functions to check if a model
as certain properties. This can allow to reduce the configuration
space for the hyperparameter optimization by only considering relevant
parameters.
"""

from ortools.sat.python import cp_model


def has_constraint_no_overlap_2d(model: cp_model.CpModel) -> bool:
    """
    Check if the model has `no_overlap_2d` constraints
    """
    return any(c.no_overlap_2d.x_intervals for c in model.proto.constraints)


def has_constraint_no_overlap(model: cp_model.CpModel) -> bool:
    """
    Check if the model has `no_overlap` constraints
    """
    return any(c.no_overlap.intervals for c in model.proto.constraints)


def has_objective(model: cp_model.CpModel) -> bool:
    """
    Check if the model has an objective
    """
    return model.has_objective()


def has_no_objective(model: cp_model.CpModel) -> bool:
    """
    Check if the model has no objective
    """
    return not model.has_objective()


class AnyOf:
    """
    A class to check if any of the provided functions return True
    """

    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, model: cp_model.CpModel) -> bool:
        return any(f(model) for f in self.funcs)
