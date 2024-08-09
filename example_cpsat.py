from pathlib import Path
from ortools.sat.python import cp_model
from google.protobuf import text_format
import optuna

def import_model(filename: str) -> cp_model.CpModel:
    model = cp_model.CpModel()
    with open(filename, "r") as file:
        text_format.Parse(file.read(), model.Proto())
    return model

import_model("models/model_proto_-8290655913605876281.pb")

models = []
for path in Path("models").rglob("*.pb"):
    models.append(import_model(str(path)))

categories = {
    "use_lns_only": [True, False],
    "repair_hint": [True, False],
    "use_lb_relax_lns": [True, False],
    "preferred_variable_order": [0, 1, 2],
    "use_erwa_heuristic": [True, False],
    "linearization_level": [0, 1, 2],
    "fp_rounding": [0,1,3,2],
    "randomize_search": [True, False],
    "diversify_lns_params": [True, False]
}

default_params = {
    "use_lns_only": False,
    "repair_hint": False,
    "use_lb_relax_lns": False,
    "preferred_variable_order": 0,
    "use_erwa_heuristic": False,
    "linearization_level": 2,
    "fp_rounding": 2,
    "randomize_search": False,
    "diversify_lns_params": False
}

def setup_solver(trial: optuna.Trial, solver: cp_model.CpSolver):
    for key, values in categories.items():
        setattr(solver.parameters, key, trial.suggest_categorical(key, values))

def objective(trial):
    obj = 0
    for model in models:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2
        #solver.parameters.relative_gap_limit = 0.01
        setup_solver(trial, solver)
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        obj += solver.objective_value
    print("Current obj", obj)
    return obj

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.enqueue_trial(default_params)
study.optimize(objective, n_trials=20)

fig = optuna.visualization.plot_param_importances(study)
fig.show()