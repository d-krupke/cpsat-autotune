import optuna
from .objective import OptunaCpSatStrategy
from .metrics import MinObjective, MaxObjective, MinTimeToOptimal
from .parameter_space import CpSatParameterSpace
from ortools.sat.python import cp_model


def print_best_params(best_params, diff_to_baseline, significant):
    print("------------------------------------------------------------")
    print("Parameters:")
    for key, value in best_params:
        print(f"\t{key}: {value}")
    print("Difference to default:", diff_to_baseline)
    print("This is significant:", significant)
    print("------------------------------------------------------------")


def _tune(
    objective: OptunaCpSatStrategy,
    parameter_space: CpSatParameterSpace,
    n_trials: int = 100,
):
    # Extract default parameter values to use in the initial trial
    default_params = parameter_space.get_default_params_for_optuna()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    study.enqueue_trial(default_params)

    study.optimize(objective, n_trials=n_trials)

    best_params, diff_to_baseline, significant = objective.best_params()
    print_best_params(best_params, diff_to_baseline, significant)
    for i in range(1, len(best_params)):
        (
            best_params_of_size,
            diff_to_baseline_of_size,
            significant_of_size,
        ) = objective.best_params(i)
        if len(best_params_of_size) == i:
            print_best_params(
                best_params_of_size, diff_to_baseline_of_size, significant_of_size
            )
    best_params = {key: value for key, value in best_params}

    return best_params, diff_to_baseline, significant


def tune_time_to_optimal(
    model: cp_model.CpModel,
    timelimit_in_s: float,
    opt_gap: float = 0.0,
    n_samples_per_param: int = 10,
    max_samples_per_param: int = 30,
    n_trials: int = 100,
):
    """
    Tune the hyperparameters of CP-SAT to minimize the time to find an optimal solution.
    The optimality condition can be relaxed by setting the optimality gap to a positive value.
    The optimality gap is considered relative to the lower bound at that time, not on the true optimal value.
    Thus, even if the solver finds the optimal solution, it might still be counted as failure if it did not find a matching lower bound.

    Args:
        model: The CP-SAT model to tune the hyperparameters for.
        timelimit_in_s: The time limit for each solve in seconds.
        opt_gap: The relative optimality gap to consider a solution optimal. If set to 0.0, only optimal solutions are considered.
        n_samples_per_param: The number of samples to take in each trial.
        max_samples_per_param: The maximum number of samples to take for each parameter. After reaching this number, the mean of the previous samples is used to improve the runtime.
        n_trials: The number of trials to run in Optuna.
    """
    parameter_space = CpSatParameterSpace()
    parameter_space.fix_parameter("use_lns_only", False)
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
    _tune(objective, parameter_space, n_trials)


def tune_for_quality_within_timelimit(
    model: cp_model.CpModel,
    timelimit_in_s: float,
    obj_for_timeout: int,
    direction: str,
    n_samples_per_param: int = 10,
    max_samples_per_param: int = 30,
    n_trials: int = 100,
):
    """
    Tune the hyperparameters of CP-SAT to get the best solution quality within a given time limit.
    This is independent of the optimality gap or lower bound but only considers the absolute objective value.

    Args:
        model: The CP-SAT model to tune the hyperparameters for.
        timelimit_in_s: The time limit for each solve in seconds.
        obj_for_timeout: The objective value to use if the solver times out. Needs to be worse than a trivial solution.
        direction: Either 'maximize' or 'minimize'. Whether to maximize or minimize the objective value.
        n_samples_per_param: The number of samples to take in each trial.
        max_samples_per_param: The maximum number of samples to take for each parameter. After reaching this number, the mean of the previous samples is used to improve the runtime.
        n_trials: The number of trials to run in Optuna.
    """
    parameter_space = CpSatParameterSpace()
    parameter_space.fix_parameter("max_time_in_seconds", timelimit_in_s)
    if direction == "maximize":
        metric = MaxObjective(obj_for_timeout)
    elif direction == "minimize":
        metric = MinObjective(obj_for_timeout)
    else:
        raise ValueError(
            f"Unknown direction {direction}. Has to be 'maximize' or 'minimize'."
        )

    objective = OptunaCpSatStrategy(
        model,
        parameter_space,
        metric=metric,
        n_samples_per_param=n_samples_per_param,
        max_samples_per_param=max_samples_per_param,
    )
    _tune(objective, parameter_space, n_trials)
