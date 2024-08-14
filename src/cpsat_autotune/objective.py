from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from ortools.sat.python import cp_model
from scipy import stats
from .parameter_space import CpSatParameterSpace
import optuna
import numpy as np
from ortools.sat import cp_model_pb2

def confidence_intervals_do_not_overlap(list1, list2, confidence=0.95):
    def calculate_confidence_interval(data, confidence):
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return mean, mean - margin_of_error, mean + margin_of_error

    # Calculate confidence intervals
    mean1, lower1, upper1 = calculate_confidence_interval(list1, confidence)
    mean2, lower2, upper2 = calculate_confidence_interval(list2, confidence)

    # Check if confidence intervals overlap
    if upper1 < lower2 or upper2 < lower1:
        return True
    else:
        return False


class Metric(ABC):
    """
    A metric that describes how good a run of the solver was. Higher is better.
    """
    @abstractmethod
    def __call__(self, status:  cp_model_pb2.CpSolverStatus, obj_value: float|None, time_in_s: float) -> float:
        pass
    
class MaxObjective(Metric):
    """
    This metric tries maximize the objective value within a time limit.
    """
    def __init__(self, obj_for_timeout: int):
        """
        Will return the objective value if a solution was found within the time limit, otherwise obj_for_timeout.
        It does not care about the status of the solver, but only if there was a feasible solution.
        :param obj_for_timeout: The value to return if the solver did not find any solution within the time limit.
        """
        self.obj_for_timeout = obj_for_timeout

    def __call__(self, status:  cp_model_pb2.CpSolverStatus, obj_value: float | None, time_in_s: float) -> float:
        if obj_value is not None:
            return obj_value
        else:
            return self.obj_for_timeout
        
class MinObjective(Metric):
    """
    Like MaxObjective, but tries to minimize the objective value within a time limit.
    Because the metric is supposed to be maximized, the value is negated internally.
    """
    def __init__(self, obj_for_timeout: int):
        self.obj_for_timeout = obj_for_timeout

    def __call__(self, status:  cp_model_pb2.CpSolverStatus, obj_value: float | None, time_in_s: float) -> float:
        if obj_value is not None:
            return -obj_value
        else:
            return -self.obj_for_timeout
        
class MinTimeToOptimal(Metric):
    """
    This metric minimizes the time it takes to find an optimal solution. Note that increasing the relative gap tolerance
    will actually consider all solutions with a gap of at most the given value as optimal.
    """
    def __init__(self, obj_for_timeout: int):
        self.obj_for_timeout = obj_for_timeout

    def __call__(self, status: cp_model_pb2.CpSolverStatus, obj_value: float | None, time_in_s: float) -> float:
        if status == cp_model.OPTIMAL:
            return -time_in_s
        else:
            return -self.obj_for_timeout

class Objective:
    def __init__(
        self,
        model: cp_model.CpModel,
        parameter_space: CpSatParameterSpace,
        metric: Metric,
        n_samples_per_param: int = 10,
        max_samples_per_param: int = 30,
    ):
        self.model = model
        self.parameter_space = parameter_space
        self.n_samples_per_param = n_samples_per_param
        self.max_samples_per_param = max_samples_per_param
        self.metric = metric
        self._baseline = []
        self._samples = defaultdict(list)

    def _solve(self, solver: cp_model.CpSolver):
        time_begin = datetime.now()
        status = solver.solve(self.model)
        time_end = datetime.now()
        solve_time = (time_end - time_begin).total_seconds()
        obj = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj = solver.objective_value
        return self.metric(status = status, obj_value=obj, time_in_s=solve_time)

    def compute_baseline(self):
        if not self._baseline:
            print("Computing baseline")
            solver = self.parameter_space.sample(None)
            values = [self._solve(solver) for _ in range(self.max_samples_per_param)]
            self._samples[frozenset()].extend(values)
            self._baseline = values
            print("Baseline:", values)
        return self._baseline

    def _get_key_from_trial(self, trial: optuna.Trial | optuna.trial.FixedTrial):
        return frozenset(self.parameter_space.get_cpsat_params_diff(trial).items())

    def __call__(self, trial: optuna.Trial):
        solver = self.parameter_space.sample(trial)
        baseline = self.compute_baseline()
        prune_if_below = min(baseline) - 0.1*(max(baseline) - min(baseline))
        param_key = self._get_key_from_trial(trial)
        n_trials = self.n_samples_per_param
        if param_key in self._samples:
            n_trials = min(
                self.max_samples_per_param - len(self._samples[param_key]), self.n_samples_per_param
            )
            n_trials = max(n_trials, 0)
        for _ in range(n_trials):
            value = self._solve(solver)
            self._samples[param_key].append(value)
            if value < prune_if_below:
                return value
        return float(np.mean(self._samples[param_key]))

    def evaluate_trial(self, trial: optuna.Trial | optuna.trial.FixedTrial):
        key = self._get_key_from_trial(trial)
        values = self._samples[key]
        baseline = self._samples[frozenset()]
        # test significance
        return np.mean(values) - np.mean(baseline), confidence_intervals_do_not_overlap(
            values, baseline
        )

    def best_params(self, max_changes: int = -1):
        keys_to_consider = [
            key for key in self._samples if max_changes < 0 or len(key) <= max_changes
        ]
        best_key = max(keys_to_consider, key=lambda x: np.mean(self._samples[x]))
        values = self._samples[best_key]
        baseline = self._samples[frozenset()]
        return (
            best_key,
            np.mean(values) - np.mean(baseline),
            confidence_intervals_do_not_overlap(values, baseline),
        )
