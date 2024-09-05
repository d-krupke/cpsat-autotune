import math
import random
from ortools.sat.python import cp_model
from cpsat_autotune import tune_time_to_optimal


def build_model() -> cp_model.CpModel:
    num_items = 100
    ratio = 0.35
    items = []
    for i in range(num_items):
        weight = random.randint(10, 1000)
        value = round(random.triangular(1, 100, 5) * weight)
        items.append((weight, value))
    capacity = math.ceil(sum(weight for weight, _ in items) * ratio)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(num_items)]
    model.add(sum(x[i] * items[i][0] for i in range(num_items)) <= capacity)
    model.maximize(sum(x[i] * items[i][1] for i in range(num_items)))
    return model


def test_tuning():
    model = build_model()
    result = tune_time_to_optimal(
        model,
        max_time_in_seconds=0.1,
        relative_gap_limit=0.01,
        n_samples_for_trial=3,
        n_samples_for_verification=5,
        n_trials=20,
    )
    assert result is not None
