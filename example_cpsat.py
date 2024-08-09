from pathlib import Path
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
    model_paths = Path(directory).rglob("*.pb")
    return [import_model(path) for path in model_paths][:1]


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
}


def setup_solver(trial: optuna.Trial, solver: cp_model.CpSolver, parameters: dict):
    for param, param_info in parameters.items():
        setattr(
            solver.parameters,
            param,
            trial.suggest_categorical(param, param_info["values"]),
        )


def objective(trial: optuna.Trial) -> float:
    total_objective_value = 0
    for model in models:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        setup_solver(trial, solver, parameter_space)

        status = solver.solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            total_objective_value += solver.objective_value

    print("Current objective value:", total_objective_value)
    return total_objective_value


# Extract default parameter values to use in the initial trial
default_params = {param: info["default"] for param, info in parameter_space.items()}

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.enqueue_trial(default_params)
study.optimize(objective, n_trials=20)

fig = optuna.visualization.plot_param_importances(study)
fig.show()
