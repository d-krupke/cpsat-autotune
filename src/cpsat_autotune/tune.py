import optuna
from .objective import Objective
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


def tune_for_quality_within_timelimit(
    model: cp_model.CpModel, timelimit_in_s: float, obj_for_timeout: int, direction: str
):
    parameter_space = CpSatParameterSpace()
    parameter_space.fix_parameter("max_time_in_seconds", timelimit_in_s)
    objective = Objective(
        model, parameter_space, direction=direction, obj_for_timeout=obj_for_timeout
    )

    # Extract default parameter values to use in the initial trial
    default_params = parameter_space.get_default_params_for_optuna()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    study.enqueue_trial(default_params)

    study.optimize(objective, n_trials=100)

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
