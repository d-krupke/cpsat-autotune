from collections import defaultdict
import numpy as np
from typing import Dict, Iterable, Union
from ortools.sat.python import cp_model

from .metrics import Metric


class ParameterEvaluator:
    """
    Evaluates the impact of parameter changes on the model's performance.
    This class identifies which parameters are essential and detects whether
    the suggested parameters are significantly better than default ones.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        params: Dict[str, Union[int, bool, float, list, tuple]],
        fixed_params: Dict[str, Union[int, bool, float, list, tuple]],
        metric: Metric,
    ) -> None:
        self.model = model
        self.params = params
        self.fixed_params = fixed_params
        self.metric = metric
        self.results = defaultdict(list)

    def _initialize_solver(
        self, params: Dict[str, Union[int, bool, float, list, tuple]]
    ) -> cp_model.CpSolver:
        """
        Initializes a solver instance with the given parameters.
        """
        solver = cp_model.CpSolver()

        for key, value in {**self.fixed_params, **params}.items():
            if isinstance(value, (list, tuple)):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)

        return solver

    def _generate_variants(
        self, params: Dict[str, Union[int, bool, float, list, tuple]]
    ) -> Iterable[tuple[str, cp_model.CpSolver]]:
        """
        Generates solver variants by iteratively excluding one parameter at a time.
        """
        for key in params:
            reduced_params = {k: v for k, v in params.items() if k != key}
            yield key, self._initialize_solver(reduced_params)

    def _evaluate_solver(self, solver: cp_model.CpSolver) -> float:
        """
        Evaluates the solver's performance using the specified metric.
        """
        return self.metric(solver, self.model)

    def evaluate(self) -> None:
        """
        Evaluates the impact of dropping each parameter individually and identifies
        the optimized set of parameters.
        """
        # Baseline evaluation with multiple runs to assess variance
        baseline_scores = [
            self._evaluate_solver(self._initialize_solver(self.params)) 
            for _ in range(10)
        ]
        baseline_avg = float(np.mean(baseline_scores))
        baseline_min = min(baseline_scores)

        print("Baseline:", self.metric.convert(baseline_avg))

        # Evaluate impact of each parameter
        optimized_params = {}
        for key, solver in self._generate_variants(self.params):
            score = self._evaluate_solver(solver)
            if score >= baseline_min:
                print(f"Drop {key}: {self.metric.convert(score)}")
            else:
                print(f"Keep {key}: {self.metric.convert(score)}")
                optimized_params[key] = self.params[key]

        # Final evaluation with optimized parameters
        optimized_score = self._evaluate_solver(self._initialize_solver(optimized_params))
        print("Optimized:", self.metric.convert(optimized_score))
        print("Optimal parameters:", optimized_params)
