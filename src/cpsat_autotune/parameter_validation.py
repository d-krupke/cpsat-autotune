from collections import defaultdict
import numpy as np
from typing import Iterable
from ortools.sat.python import cp_model

from .metrics import Metric


class ParameterValidation:
    """
    This class tries to check the influence of the parameter changes on the model.
    The best parameters can contain a lot of random deviations from the default parameters due to the search process.
    This class tries to find out, which parameters are actually important, and maybe even detect if the suggested
    parameters were just a lucky shot.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        params: dict[str, int | bool | float | list | tuple],
        fixed_params: dict[str, int | bool | float | list | tuple],
        metric: Metric,
    ) -> None:
        self.model = model
        self.params = params
        self.fixed_params = fixed_params
        self.values = defaultdict(list)
        self.metric = metric

    def _create_solver(
        self, params: dict[str, int | bool | float | list | tuple]
    ) -> cp_model.CpSolver:
        """
        Create a solver with the given parameters.
        """
        solver = cp_model.CpSolver()
        for key, value in self.fixed_params.items():
            if isinstance(value, list | tuple):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)
        for key, value in params.items():
            if key in self.fixed_params:
                continue
            if isinstance(value, list | tuple):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)
        return solver

    def _iter_drop_one(
        self, params: dict[str, int | bool | float | list | tuple]
    ) -> Iterable[tuple[str, cp_model.CpSolver]]:
        """
        Iterate over all parameters and return a solver with all parameters except one.
        """
        for key in params:
            new_params = params.copy()
            new_params.pop(key)
            yield key, self._create_solver(new_params)

    def _evaluate(self, solver: cp_model.CpSolver) -> float:
        """
        Evaluate the solver with the model.
        """
        return self.metric(solver, self.model)

    def evaluate(self):
        """
        This function evaluates the performance when dropping one parameter at a time.
        """
        # baseline
        # We do multiple runs of the baseline to get some idea of the variance.
        baseline_values = [
            self._evaluate(self._create_solver(self.params)) for _ in range(10)
        ]
        baseline_value = float(np.mean(baseline_values))
        lb = min(baseline_values)
        print("Baseline:", self.metric.convert(baseline_value))
        # all parameters
        opt_params = {}
        for key, solver in self._iter_drop_one(self.params):
            value = self._evaluate(solver)
            if value >= lb:
                # There is a good chance that this parameter is not important.
                # This is rather aggressive, but one can usually assume that the
                #  default parameters are already good, so being eager in returning to the default should
                # be less risky than deviating from the default parameters.
                print("Drop", key, ":", self.metric.convert(value))
            else:
                print("Keep", key, ":", self.metric.convert(value))
                opt_params[key] = self.params[key]
        optimized_value = self._evaluate(self._create_solver(opt_params))
        print("Optimized:", self.metric.convert(optimized_value))
        print("Optimal parameters:", opt_params)
