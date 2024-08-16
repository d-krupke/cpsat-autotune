from dataclasses import dataclass
import numpy as np
from ortools.sat.python import cp_model
from .metrics import Comparison, Metric


@dataclass
class MultiResult:
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
        return MultiResult(params=self.params, scores=[metric.worst(self)]*len(self.scores))
    
    def __repr__(self) -> str:
        return f"MultiResult(scores={self.scores}, params={self.params})"


class CachingScorer:
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
        return param_set

    def _remove_fixed_params(
        self, params: dict[str, float | int | bool | list | tuple]
    ) -> dict[str, float | int | bool | list | tuple]:
        return {
            key: value for key, value in params.items() if key not in self.fixed_params
        }

    def _prepare_solver(
        self, params: dict[str, float | int | bool | list | tuple]
    ) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)
        for key, value in self.fixed_params.items():
            if isinstance(value, (list, tuple)):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)
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
        params = self._remove_fixed_params(params)
        param_key: frozenset = self._create_key_from_params(params)
        result = self._cache.get(param_key, MultiResult(scores=[], params=params))
        if len(result) >= num_runs:
            return result
        if knockout_score is not None and len(result) > 0:
            if self.metric.comp(self.metric.worst(result), knockout_score) in (Comparison.WORSE, Comparison.EQUAL):
                print("Returning cached knockout result")
                return result.as_knockout_result(self.metric)
        n_missing = num_runs - len(result)
        for _ in range(n_missing):
            solver = self._prepare_solver(params)
            score = self.metric(solver, self.model)
            result.scores.append(score)
            if knockout_score is not None:
                if self.metric.comp(score, knockout_score) in (Comparison.WORSE, Comparison.EQUAL):
                    self._cache[param_key] = result
                    print("Returning knockout result")
                    return result.as_knockout_result(self.metric)
        self._cache[param_key] = result
        return result

    def __iter__(self):
        return iter(self._cache.values())