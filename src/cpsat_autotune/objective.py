from collections import defaultdict
from ortools.sat.python import cp_model
from scipy import stats
from .parameter_space import CpSatParameterSpace
import optuna
import numpy as np


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


class Objective:
    def __init__(
        self,
        model: cp_model.CpModel,
        parameter_space: CpSatParameterSpace,
        direction: str = "maximize",
        obj_for_timeout: int = 0,
    ):
        if direction not in ("maximize", "minimize"):
            raise ValueError("Direction must be 'maximize' or 'minimize'")
        self.maximization = direction == "maximize"
        self.model = model
        self.parameter_space = parameter_space
        self.n_trials = 10
        self.max_trials = 30
        self.obj_for_timeout = 0
        self._baseline = []
        self._samples = defaultdict(list)

    def _solve(self, solver: cp_model.CpSolver):
        status = solver.solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return solver.objective_value * (1 if self.maximization else -1)
        else:
            return self.obj_for_timeout * (1 if self.maximization else -1)

    def compute_baseline(self):
        if not self._baseline:
            print("Computing baseline")
            solver = self.parameter_space.sample(None)
            values = [self._solve(solver) for _ in range(self.max_trials)]
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
        n_trials = self.n_trials
        if param_key in self._samples:
            n_trials = min(
                self.max_trials - len(self._samples[param_key]), self.n_trials
            )
            n_trials = max(n_trials, 0)
        for _ in range(n_trials):
            value = self._solve(solver)
            self._samples[param_key].append(value)
            if value < prune_if_below:
                raise optuna.TrialPruned()
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
