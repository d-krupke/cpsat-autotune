from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Iterable, Union
from ortools.sat.python import cp_model

from .cpsat_parameters import get_parameter_by_name

from .metrics import Metric

def log(message: str) -> None:
    """
    Logs a message to the console.
    """
    print(message)

@dataclass
class EvaluationResult:
    """
    Data class that stores the results of the parameter evaluation.
    """
    optimized_params: Dict[str, Union[int, bool, float, list, tuple]]
    contribution: Dict[str, float]
    default_score: float
    optimized_score: float

    def print_results(self) -> None:
        """
        Prints the evaluation results in a professional format.
        """
        print("============================================================")
        print("                 OPTIMIZED PARAMETERS")
        print("============================================================")
        for i, (key, value) in enumerate(self.optimized_params.items(), start=1):
            default_value = get_parameter_by_name(key).get_cpsat_default()
            description = get_parameter_by_name(key).description.strip().replace('\n', '\n\t\t')
            contribution_value = f"{self.contribution[key]:.2%}" if key in self.contribution else "<NA>"
            print(f"\n{i}. {key}: {value}")
            print(f"\tContribution: {contribution_value}")
            print(f"\tDefault Value: {default_value}")
            print(f"\tDescription:\n\t\t{description}")

        print("------------------------------------------------------------")
        print(f"Default Metric Value: {self.default_score}")
        print(f"Optimized Metric Value: {self.optimized_score}")
        print("------------------------------------------------------------")
        
        print("\n============================================================")
        print("*** WARNING ***")
        print("============================================================")
        print(
            "The optimized parameters listed above were obtained based on a sampling approach "
            "and may not fully capture the complexities of the entire problem space. "
            "While statistical reasoning has been applied, these results should be considered "
            "as a suggestion for further evaluation rather than definitive settings.\n"
            "It is strongly recommended to validate these parameters in larger, more comprehensive "
            "experiments before adopting them in critical applications."
        )
        print("============================================================")


class ParameterEvaluator:
    """
    Evaluates the impact of parameter changes on the model's performance.
    Identifies essential parameters and determines whether suggested parameters
    offer significant improvements over the defaults.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        params: Dict[str, Union[int, bool, float, list, tuple]],
        fixed_params: Dict[str, Union[int, bool, float, list, tuple]],
        metric: Metric,
        default_score: float,
        baseline_values: list[float] | None = None,
        min_baseline_size: int = 10,
    ) -> None:
        self.model = model
        self.params = params
        self.fixed_params = fixed_params
        self.metric = metric
        self.results = defaultdict(list)
        self.baseline_scores = baseline_values if baseline_values else []
        self.min_baseline_size = min_baseline_size
        self.default_score = default_score

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

    def evaluate(self) -> EvaluationResult:
        """
        Evaluates the impact of excluding each parameter individually, identifies
        the optimized set of parameters, and returns an EvaluationResult object
        containing the results.
        """
        log("Checking which parameter changes obtained by the hyperparameter optimization are essential...")
        # Baseline evaluation with multiple runs to assess variance
        while len(self.baseline_scores) < self.min_baseline_size:
            self.baseline_scores.append(
                self._evaluate_solver(self._initialize_solver(self.params))
            )
        baseline_avg = float(np.mean(self.baseline_scores))
        worst_baseline = min(self.baseline_scores)

        optimized_params = {}
        diffs = {}

        for key, solver in self._generate_variants(self.params):
            log(f"Evaluating resetting parameter '{key}' to default..." )
            score = self._evaluate_solver(solver)
            if score >= worst_baseline:
                log(f"Seems like we can drop parameter '{key}' from the optimized parameters.")
                continue
            else:
                log(f"The parameter '{key}' seems to be essential for the performance.")
                optimized_params[key] = self.params[key]
                diffs[key] = abs(baseline_avg - score)

        # Calculate parameter significance
        total_diff = sum(diffs.values())
        significance = {key: diff / total_diff for key, diff in diffs.items()}

        # Final evaluation with optimized parameters
        optimized_score = self._evaluate_solver(self._initialize_solver(optimized_params))

        if optimized_score < worst_baseline:
            # revert to initial parameters as the seems to be some difficult correlations
            log("The final evaluation indicates that dropping all of the seemingly uninfluential parameters did worsen the performance. Reverting to the initial parameters.")
            optimized_params = self.params
            optimized_score = baseline_avg
            significance = {}

        # Convert the metrics before storing them in the result
        return EvaluationResult(
            optimized_params=optimized_params,
            contribution=significance,
            default_score=self.metric.convert(self.default_score),
            optimized_score=self.metric.convert(optimized_score),
        )
