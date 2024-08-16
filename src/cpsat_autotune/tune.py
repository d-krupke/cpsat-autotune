import optuna

from .caching_solver import CachingScorer, MultiResult

from .objective import OptunaCpSatStrategy
from .metrics import Metric, MinObjective, MaxObjective, MinTimeToOptimal
from .parameter_space import CpSatParameterSpace
from ortools.sat.python import cp_model
from .parameter_evaluator import ParameterEvaluator


def _tune(
    parameter_space: CpSatParameterSpace,
    model: cp_model.CpModel,
    metric: Metric,
    max_samples_per_param: int,
    n_samples_per_param: int,
    n_trials: int = 100,
) -> MultiResult:
    """
    Perform hyperparameter tuning using Optuna.

    Args:
        objective: An instance of OptunaCpSatStrategy defining the objective for tuning.
        parameter_space: An instance of CpSatParameterSpace defining the search space for parameters.
        n_trials: The number of trials to execute in the tuning process.

    Returns:
        The best parameters found during the tuning process.
    """
    scorer = CachingScorer(model, metric)

    default_baseline = scorer.evaluate({}, max_samples_per_param)

    print(f"Baseline: min={default_baseline.min()}, mean={default_baseline.mean()}, max={default_baseline.max()}")

    objective = OptunaCpSatStrategy(
        parameter_space,
        scorer=scorer,
        n_samples_for_trial=n_samples_per_param,
        n_samples_for_verification=max_samples_per_param,
    )

    
    # Initialize the study with the given parameter space and objective
    default_params = parameter_space.get_default_params_for_optuna()
    study = optuna.create_study(
        direction=objective.scorer.metric.direction, sampler=optuna.samplers.TPESampler()
    )
    study.enqueue_trial(default_params)

    # Optimize the study with the objective function
    study.optimize(objective, n_trials=n_trials)

    # Retrieve and print the best parameters
    best_params = objective.best_params()
    print(f"Best parameters: {best_params.params}. Score: {best_params.mean()}")
    # _print_best_params(best_params, diff_to_baseline, significant)

    # Return the best parameters and performance results
    be = ParameterEvaluator(
        params=best_params.params,
        scorer=scorer,
        metric=metric,
    )
    result = be.evaluate()
    result.print_results(default_baseline.mean())
    return best_params


def tune_time_to_optimal(
    model: cp_model.CpModel,
    timelimit_in_s: float,
    opt_gap: float = 0.0,
    n_samples_per_param: int = 10,
    max_samples_per_param: int = 30,
    n_trials: int = 100,
) -> dict:
    """
    Tune CP-SAT hyperparameters to minimize the time required to find an optimal solution.

    Args:
        model: The CP-SAT model for which the hyperparameters are tuned.
        timelimit_in_s: The time limit for each solve operation in seconds.
        opt_gap: The relative optimality gap that determines when a solution is considered optimal.
                 A value of 0.0 requires the solution to be exactly optimal.
        n_samples_per_param: The number of samples per parameter to take in each trial.
        max_samples_per_param: The maximum number of samples per parameter allowed before using the mean to improve runtime.
        n_trials: The number of trials to execute in the tuning process.
    """
    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("use_lns_only")  # never useful for this metric
    parameter_space.drop_parameter("max_time_in_seconds")
    if opt_gap > 0.0:
        parameter_space.drop_parameter("relative_gap_tolerance")

    metric = MinTimeToOptimal(max_time_in_seconds=timelimit_in_s, relative_gap_limit=opt_gap)

    return _tune(
        parameter_space=parameter_space,
        model=model,
        metric=metric,
        max_samples_per_param=max_samples_per_param,
        n_samples_per_param=n_samples_per_param,
        n_trials=n_trials
    ).params


def tune_for_quality_within_timelimit(
    model: cp_model.CpModel,
    timelimit_in_s: float,
    obj_for_timeout: int,
    direction: str,
    n_samples_per_param: int = 10,
    max_samples_per_param: int = 30,
    n_trials: int = 100,
) -> dict:
    """
    Tune CP-SAT hyperparameters to maximize or minimize solution quality within a given time limit.

    Args:
        model: The CP-SAT model for which the hyperparameters are tuned.
        timelimit_in_s: The time limit for each solve operation in seconds.
        obj_for_timeout: The objective value to apply if the solver times out. Should be worse than a trivial solution.
        direction: A string specifying whether to 'maximize' or 'minimize' the objective value.
        n_samples_per_param: The number of samples per parameter to take in each trial.
        max_samples_per_param: The maximum number of samples per parameter allowed before using the mean to improve runtime.
        n_trials: The number of trials to execute in the tuning process.

    Raises:
        ValueError: If the `direction` argument is not 'maximize' or 'minimize'.
    """
    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("max_time_in_seconds")

    if direction == "maximize":
        metric = MaxObjective(obj_for_timeout=obj_for_timeout, max_time_in_seconds=timelimit_in_s)
    elif direction == "minimize":
        metric = MinObjective(obj_for_timeout=obj_for_timeout, max_time_in_seconds=timelimit_in_s)
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'maximize' or 'minimize'."
        )
    return _tune(
        parameter_space=parameter_space,
        model=model,
        metric=metric,
        max_samples_per_param=max_samples_per_param,
        n_samples_per_param=n_samples_per_param,
        n_trials=n_trials
    ).params
