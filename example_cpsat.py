from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
from google.protobuf import text_format
import optuna


def import_model(filepath: Path) -> cp_model.CpModel:
    model = cp_model.CpModel()
    with filepath.open("r") as file:
        text_format.Parse(file.read(), model.Proto())
    return model


# Automatically load all models from the 'models' directory
def load_models(directory: str) -> list:
    # model_paths = Path(directory).rglob("*.pb")
    model_paths = [Path("models/model_proto_-84479006250281510.pb")]
    return [import_model(path) for path in model_paths]


models = load_models("models")

# These are the parameters of CP-SAT solver that we want to optimize
parameter_space = {
    "use_lns_only": {"values": [True, False], "default": False},
    "repair_hint": {"values": [True, False], "default": False},
    "use_lb_relax_lns": {"values": [True, False], "default": False},
    "preferred_variable_order": {"values": [0, 1, 2], "default": 0},
    "use_erwa_heuristic": {"values": [True, False], "default": False},
    "linearization_level": {"values": [0, 1, 2], "default": 1},
    "fp_rounding": {"values": [0, 1, 3, 2], "default": 2},
    "randomize_search": {"values": [True, False], "default": False},
    "diversify_lns_params": {"values": [True, False], "default": False},
    "add_objective_cut": {"values": [True, False], "default": False},
    "use_objective_lb_search": {"values": [True, False], "default": False},
    "use_objective_shaving_search": {"values": [True, False], "default": False},
    "search_branching": {"values": [0, 1, 2, 3, 4, 5, 6, 7, 8], "default": 0},
    # "polarity_rephase_increment": {"values": [500, 1000, 2000], "default": 1000},
    # "random_polarity_ratio": {"values": [0.0, 0.1, 0.2, 0.5], "default": 0.0},
    # "random_branches_ratio": {"values": [0.0, 0.1, 0.2, 0.5], "default": 0.0},
    # "minimization_algorithm": {"values": [0, 1, 2, 3], "default": 2},
    # "restart_algorithms": {"values": [0, 1, 2, 3, 4], "default": 1},
    "cut_level": {"values": [0, 1], "default": 1},
    "max_all_diff_cut_size": {"values": [32, 64, 128], "default": 64},
    "symmetry_level": {"values": [0, 1, 2], "default": 2},
    "max_presolve_iterations": {"values": [1, 2, 3, 5, 10], "default": 3},
    #"cp_model_presolve": {"values": [True, False], "default": True},
    "cp_model_probing_level": {"values": [0, 1, 2], "default": 2},
    "presolve_probing_deterministic_time_limit": {
        "values": [5.0, 10.0, 30.0],
        "default": 30.0,
    },
    "presolve_bve_threshold": {"values": [100, 500, 1000], "default": 500},
}


def setup_solver(
    trial: optuna.Trial | None | dict, solver: cp_model.CpSolver, parameters: dict
):
    if trial is None:
        return
    if isinstance(trial, dict):
        for param, value in trial.items():
            setattr(solver.parameters, param, value)
        return
    for param, param_info in parameters.items():
        setattr(
            solver.parameters,
            param,
            trial.suggest_categorical(param, param_info["values"]),
        )


baseline = 0


def objective(trial: optuna.Trial | None | dict, num_repeats: int = 10) -> float:
    total_objective_value = 0
    for model in models:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1
        setup_solver(trial, solver, parameter_space)

        objs = []
        for i in range(num_repeats):
            status = solver.solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                objs.append(solver.objective_value)
            else:
                objs.append(0)
        total_objective_value += np.median(objs)

    print("Current objective value:", total_objective_value)
    return total_objective_value - baseline


# Calculate the baseline objective value
baseline = objective(None)

# Extract default parameter values to use in the initial trial
default_params = {param: info["default"] for param, info in parameter_space.items()}

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.enqueue_trial(default_params)
study.optimize(objective, n_trials=100)

# show differences of best trial to default parameters
print("Best trial:")
for key, value in study.best_trial.params.items():
    if value != default_params[key]:
        print(f"{key}: {value}")

# Do a t-test to determine if the difference is significant
N = 50
baseline_results = [objective(None, num_repeats=1) for _ in range(N)]
best_trial_results = [
    objective(study.best_trial.params, num_repeats=1) for _ in range(N)
]

from scipy.stats import ttest_rel

t_statistic, p_value = ttest_rel(baseline_results, best_trial_results)
print(f"p-value: {p_value}")
print("Median baseline:", np.median(baseline_results))
print("Median best trial:", np.median(best_trial_results))

# Plot the parameter importances

fig = optuna.visualization.plot_param_importances(study)
fig.show()
