import optuna

from .objective import OptunaCpSatStrategy, ParameterStats
from .metrics import MinObjective, MaxObjective, MinTimeToOptimal
from .parameter_space import CpSatParameterSpace
from ortools.sat.python import cp_model
from .parameter_evaluator import ParameterEvaluator


def _tune(
    objective: OptunaCpSatStrategy,
    parameter_space: CpSatParameterSpace,
    n_trials: int = 100,
) -> ParameterStats:
    """
    Perform hyperparameter tuning using Optuna.

    Args:
        objective: An instance of OptunaCpSatStrategy defining the objective for tuning.
        parameter_space: An instance of CpSatParameterSpace defining the search space for parameters.
        n_trials: The number of trials to execute in the tuning process.

    Returns:
        The best parameters found during the tuning process.
    """
    # Initialize the study with the given parameter space and objective
    default_params = parameter_space.get_default_params_for_optuna()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    study.enqueue_trial(default_params)

    # Optimize the study with the objective function
    study.optimize(objective, n_trials=n_trials)

    # Retrieve and print the best parameters
    best_stats = objective.best_params()
    print(best_stats.as_text())
    # _print_best_params(best_params, diff_to_baseline, significant)

    # Evaluate parameter subsets and print if relevant
    for i in range(1, best_stats.changes):
        best_stats = objective.best_params(i)
        if best_stats.changes == i:
            print(best_stats.as_text())
    print("Baseline:")
    print(objective.get_baseline().as_text())
    # Return the best parameters and performance results
    return best_stats


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
    parameter_space.fix_parameter("use_lns_only", False)  # never useful for this metric
    parameter_space.fix_parameter("max_time_in_seconds", timelimit_in_s)
    if opt_gap > 0.0:
        parameter_space.fix_parameter("relative_gap_tolerance", opt_gap)

    metric = MinTimeToOptimal(obj_for_timeout=int(10 * timelimit_in_s))
    objective = OptunaCpSatStrategy(
        model,
        parameter_space,
        metric=metric,
        n_samples_per_param=n_samples_per_param,
        max_samples_per_param=max_samples_per_param,
    )

    stats = _tune(objective, parameter_space, n_trials)
    values = [metric.convert_to_maximization(v) for v in stats.values]
    be = ParameterEvaluator(
        model=model,
        params=stats.cpsat_params,
        default_score=metric.convert_to_maximization(objective.get_baseline().mean),
        metric=metric,
        fixed_params=parameter_space.fixed_parameters,
        baseline_values=values,
    )
    result = be.evaluate()
    result.print_results()
    return result.optimized_params


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
    parameter_space.fix_parameter("max_time_in_seconds", timelimit_in_s)

    if direction == "maximize":
        metric = MaxObjective(obj_for_timeout)
    elif direction == "minimize":
        metric = MinObjective(obj_for_timeout)
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'maximize' or 'minimize'."
        )

    objective = OptunaCpSatStrategy(
        model,
        parameter_space,
        metric=metric,
        n_samples_per_param=n_samples_per_param,
        max_samples_per_param=max_samples_per_param,
    )

    stats = _tune(objective, parameter_space, n_trials)
    values = [metric.convert_to_maximization(v) for v in stats.values]
    be = ParameterEvaluator(
        model=model,
        params=stats.cpsat_params,
        default_score=metric.convert_to_maximization(objective.get_baseline().mean),
        metric=metric,
        fixed_params=parameter_space.fixed_parameters,
        baseline_values=values,
    )
    result = be.evaluate()
    result.print_results()
    return result.optimized_params
