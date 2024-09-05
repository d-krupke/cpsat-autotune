import logging
from dataclasses import dataclass
import numpy as np
from ortools.sat.python import cp_model

from cpsat_autotune.cpsat_parameters import get_parameter_by_name
from .metrics import Comparison, Metric
from ortools.sat import sat_parameters_pb2

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler()  # You can add more handlers (e.g., file handlers) as needed
    ],
)


@dataclass
class MultiResult:
    """
    Instead of just the mean score, we store all samples to compute additional statistics.
    """

    scores: list[float]
    params: dict[str, float | int | bool | list | tuple]

    def mean(self) -> float:
        return sum(self.scores) / len(self.scores)

    def median(self) -> float:
        return float(np.median(self.scores))

    def std(self) -> float:
        return float(np.std(self.scores))

    def max(self) -> float:
        return float(np.max(self.scores))

    def min(self) -> float:
        return float(np.min(self.scores))

    def spread(self) -> float:
        return self.max() - self.min()

    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self):
        return iter(self.scores)

    def as_knockout_result(self, metric: Metric) -> "MultiResult":
        return MultiResult(
            params=self.params, scores=[metric.worst(self)] * len(self.scores)
        )

    def __repr__(self) -> str:
        return "MultiResult(scores=%s, params=%s)" % (self.scores, self.params)


class CachingScorer:
    """
    Computing the score for a given set of parameters involves running the solver multiple times.
    As this can be computationally expensive, the results are cached to avoid redundant computations.
    This can also be used later to get better statistics about the performance of the parameters.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        metric: Metric,
        fixed_params: dict[str, float | int | bool | list | tuple] | None = None,
    ) -> None:
        self.model = model
        self.metric = metric
        self._cache: dict[frozenset, MultiResult] = {}
        self.fixed_params = (
            fixed_params.copy() if fixed_params else {}
        )  # DO NOT MODIFY THIS DICTIONARY
        self.direction = metric.direction

    def _create_key_from_params(
        self, params: dict[str, float | int | bool | list | tuple]
    ) -> frozenset:
        def _replace_lists(value):
            if isinstance(value, list):
                return tuple(sorted(value))
            if isinstance(value, tuple):
                return tuple(sorted(value))
            return value

        param_set = frozenset(
            (key, _replace_lists(value)) for key, value in params.items()
        )
        logging.debug("Created key from params: %s", param_set)
        return param_set

    def _remove_fixed_params(
        self, params: dict[str, float | int | bool | list | tuple]
    ) -> dict[str, float | int | bool | list | tuple]:
        cleaned_params = {
            key: value for key, value in params.items() if key not in self.fixed_params
        }
        logging.debug("Removed fixed params: %s", cleaned_params)
        return cleaned_params

    def _prepare_solver(
        self, params: dict[str, float | int | bool | list | tuple]
    ) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        subsolver = sat_parameters_pb2.SatParameters()
        subsolver.name = "tuned_solver"
        has_subsolver_params = False
        for key, value in params.items():
            is_subsolver_param = get_parameter_by_name(key).subsolver
            level = subsolver if is_subsolver_param else solver.parameters
            if is_subsolver_param:
                has_subsolver_params = True
            if isinstance(value, (list, tuple)):
                getattr(level, key).extend(value)
            else:
                setattr(level, key, value)
        for key, value in self.fixed_params.items():
            is_subsolver_param = get_parameter_by_name(key).subsolver
            level = subsolver if is_subsolver_param else solver.parameters
            if is_subsolver_param:
                has_subsolver_params = True
            if isinstance(value, (list, tuple)):
                getattr(level, key).extend(value)
            else:
                setattr(level, key, value)
        if has_subsolver_params:
            solver.parameters.subsolver_params.append(subsolver)
            solver.parameters.extra_subsolvers.append(subsolver.name)
        logging.debug("Solver prepared with params: %s", params)
        return solver

    def evaluate(
        self,
        params: dict[str, float | int | bool | list | tuple],
        num_runs: int = 1,
        knockout_score: float | None = None,
    ) -> MultiResult:
        """
        Args:
            params: The parameters to evaluate.
            num_runs: The number of runs to average the score over.
            knockout_score: Abort early if the score is worse than this value.
        """
        logging.info(
            "Evaluating with params: %s, num_runs: %s, knockout_score: %s",
            params,
            num_runs,
            knockout_score,
        )
        params = self._remove_fixed_params(params)
        param_key: frozenset = self._create_key_from_params(params)
        result = self._cache.get(param_key, MultiResult(scores=[], params=params))
        if len(result) >= num_runs:
            logging.info("Returning cached result.")
            return result
        if knockout_score is not None and len(result) > 0:
            worst_score = self.metric.worst(result)
            if self.metric.comp(worst_score, knockout_score) in (
                Comparison.WORSE,
                Comparison.EQUAL,
            ):
                logging.info("Returning cached knockout result.")
                return result.as_knockout_result(self.metric)
        n_missing = num_runs - len(result)
        for _ in range(n_missing):
            solver = self._prepare_solver(params)
            score = self.metric(solver, self.model)
            result.scores.append(score)
            logging.debug("Run completed with score: %s", score)
            if knockout_score is not None:
                if self.metric.comp(score, knockout_score) in (
                    Comparison.WORSE,
                    Comparison.EQUAL,
                ):
                    self._cache[param_key] = result
                    logging.info("Returning knockout result.")
                    return result.as_knockout_result(self.metric)
        self._cache[param_key] = result
        logging.info("Evaluation completed and result cached.")
        return result

    def __iter__(self):
        logging.debug("Iterating over cached results.")
        return iter(self._cache.values())
