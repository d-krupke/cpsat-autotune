from collections import defaultdict
from datetime import datetime
import random
from ortools.sat.python import cp_model
from scipy import stats
from .parameter_space import CpSatParameterSpace
import optuna
import numpy as np
from .metrics import Metric


def do_cis_overlap(list1, list2, confidence=0.95):
    """
    Check if the confidence intervals of two lists overlap.
    """

    def calculate_confidence_interval(data, confidence):
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return mean - margin_of_error, mean + margin_of_error

    # Calculate confidence intervals
    lower1, upper1 = calculate_confidence_interval(list1, confidence)
    lower2, upper2 = calculate_confidence_interval(list2, confidence)

    # Check if confidence intervals overlap
    return not (upper1 < lower2 or upper2 < lower1)

class ParameterStats:
    def __init__(self, cpsat_params: dict, values: list[float]):
        self.cpsat_params = cpsat_params
        self.values = values
        self.mean = float(np.mean(values))
        self.median = float(np.median(values))
        self.std = float(np.std(values))
        self.max = float(np.max(values))
        self.min = float(np.min(values))
        self.count = len(values)
        self.changes = len(cpsat_params)

    def do_cis_overlap(self, other, confidence=0.95):
        return do_cis_overlap(self.values, other.values, confidence)
    
    def as_text(self):
        text = "Parameter Statistics:\n"
        if not self.cpsat_params:
            text += "\tDefault Parameters\n"
        else:
            for key, value in self.cpsat_params.items():
                text += f"\t{key}: {value}\n"
        text += f" Mean: {self.mean}\n"
        text += f" Median: {self.median}\n"
        text += f" Standard Deviation: {self.std}\n"
        text += f" Max: {self.max}\n"
        text += f" Min: {self.min}\n"
        text += f" Count: {self.count}\n"
        return text

class OptunaCpSatStrategy:
    """
    This class interacts with Optuna to tune the hyperparameters of a CP-SAT model.
    """

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
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        time_begin = datetime.now()
        status = solver.solve(self.model)
        time_end = datetime.now()
        solve_time = (time_end - time_begin).total_seconds()
        obj = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj = solver.objective_value
        return self.metric(status=status, obj_value=obj, time_in_s=solve_time)

    def compute_baseline(self):
        """
        Compute the baseline by solving the model with the default parameters.
        """
        if not self._baseline:
            print("Computing baseline")
            solver = self.parameter_space.sample(None)
            values = [self._solve(solver) for _ in range(self.max_samples_per_param)]
            self._samples[frozenset()].extend(values)
            self._baseline = values
            print("Baseline:", values)
        return self._baseline

    def _get_key_from_trial(self, trial: optuna.Trial | optuna.trial.FixedTrial):
        """
        Returns a hashable frozen set of the parameters that are different from the default parameters.
        """
        return frozenset(self.parameter_space.get_cpsat_params_diff(trial).items())

    def __call__(self, trial: optuna.Trial):
        """
        This function is called by Optuna to evaluate a trial.
        """
        solver = self.parameter_space.sample(trial)
        baseline = self.compute_baseline()
        prune_if_below = min(baseline) - 0.1 * (max(baseline) - min(baseline))
        param_key = self._get_key_from_trial(trial)
        n_trials = self.n_samples_per_param
        if param_key in self._samples:
            n_trials = min(
                self.max_samples_per_param - len(self._samples[param_key]),
                self.n_samples_per_param,
            )
            n_trials = max(n_trials, 0)
        for _ in range(n_trials):
            value = self._solve(solver)
            self._samples[param_key].append(value)
            if value < prune_if_below:
                # Save some time by pruning the trial if it is already worse than the baseline
                # Instead of using Optuna's pruner, we return the bad value as this allows
                # Optuna to still use the value for the study. We just don't waste time on
                # getting a more precise measurement.
                return value
        return float(np.mean(self._samples[param_key]))

    def evaluate_trial(self, trial: optuna.Trial | optuna.trial.FixedTrial) -> ParameterStats:
        key = self._get_key_from_trial(trial)
        values = self._samples[key]
        return ParameterStats(dict(key), [self.metric.convert(v) for v in values])

    def get_baseline(self) -> ParameterStats:
        return ParameterStats({}, [self.metric.convert(v) for v in self.compute_baseline()])

    def best_params(self, max_changes: int = -1) -> ParameterStats:
        """
        Returns the best parameters found so far. Use this function instead of the Optuna study's best_params
        function as it not only converts the parameters to the actual CP-SAT parameters, but can also give
        more information about the significance of the results.
        """
        keys_to_consider = [
            key for key in self._samples if max_changes < 0 or len(key) <= max_changes
        ]
        best_key = max(keys_to_consider, key=lambda x: np.mean(self._samples[x]))
        return ParameterStats(dict(best_key), [self.metric.convert(v) for v in  self._samples[best_key]])
