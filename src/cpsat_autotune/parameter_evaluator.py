from collections import defaultdict
import numpy as np
from typing import Dict, Iterable, Union
from ortools.sat.python import cp_model

from .metrics import Metric


class ParameterEvaluator:
    """
    Evaluates the impact of parameter changes on the model's performance.
    This class identifies which parameters are essential and determines whether
    the suggested parameters offer significant improvements over the defaults.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        params: Dict[str, Union[int, bool, float, list, tuple]],
        fixed_params: Dict[str, Union[int, bool, float, list, tuple]],
        metric: Metric,
        baseline_values: list[float]|None = None,
        min_baseline_size: int = 10,
    ) -> None:
        self.model = model
        self.params = params
        self.fixed_params = fixed_params
        self.metric = metric
        self.results = defaultdict(list)
        self.baseline_scores = baseline_values if baseline_values else []
        self.min_baseline_size = min_baseline_size 

    def _initialize_solver(
        self, params: Dict[str, Union[int, bool, float, list, tuple]]
    ) -> cp_model.CpSolver:
        """
        Initializes a solver instance with the specified parameters.
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
        Generates solver variants by excluding one parameter at a time.
        """
        for key in params:
            reduced_params = {k: v for k, v in params.items() if k != key}
            yield key, self._initialize_solver(reduced_params)

    def _evaluate_solver(self, solver: cp_model.CpSolver) -> float:
        """
        Evaluates the solver's performance using the provided metric.
        """
        return self.metric(solver, self.model)

    def evaluate(self) -> Dict[str, Union[int, bool, float, list, tuple]]:
        """
        Evaluates the impact of excluding each parameter individually, identifies
        the optimized set of parameters, and returns the optimal parameters.
        """
        # Baseline evaluation with multiple runs to assess variance
        while len(self.baseline_scores) < self.min_baseline_size:
            self.baseline_scores.append(
                self._evaluate_solver(self._initialize_solver(self.params))
            )
        baseline_avg = float(np.mean(self.baseline_scores))
        worst_baseline = min(self.baseline_scores)  # worst performance in the baseline

        print("Baseline:", self.metric.convert(baseline_avg))
        print("Dropping all parameter that are not required to be better than", self.metric.convert(worst_baseline))

        # Evaluate the impact of each parameter
        optimized_params = {}
        diffs = {}

        for key, solver in self._generate_variants(self.params):
            score = self._evaluate_solver(solver)
            if score >= worst_baseline: 
                # performance is still not worse than the worst baseline
                # so probably the parameter is not essential
                print(f"\tDrop {key}: {self.metric.convert(score)}")
            else:
                print(f"\tKeep {key}: {self.metric.convert(score)}")
                optimized_params[key] = self.params[key]
                diffs[key] = abs(baseline_avg - score)

        # Calculate and display parameter significance
        total_diff = sum(diffs.values())
        significance = [(key, diff / total_diff) for key, diff in diffs.items()]
        significance.sort(key=lambda x: x[1], reverse=True)

        print("Parameter Significance:")
        for key, sig in significance:
            print(f"\t{key}: {sig:.2%}")

        # Final evaluation with optimized parameters
        optimized_score = self._evaluate_solver(self._initialize_solver(optimized_params))
        print("Optimized:", self.metric.convert(optimized_score))
        print("Optimal Parameters:", optimized_params)

        # Return the most suitable parameters
        return optimized_params if optimized_score > worst_baseline else self.params
