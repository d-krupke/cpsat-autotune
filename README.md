# cpsat-autotune: A Hyperparameter Tuning Tool for Google's OR-Tools CP-SAT Solver

**cpsat-autotune** is a Python library designed to optimize the hyperparameters of Google's OR-Tools CP-SAT solver for specific problem instances. While CP-SAT is already highly optimized for a broad range of generic problems, fine-tuning its parameters for particular problem sets can yield significant performance gains. This tool leverages the `optuna` optimization library to systematically explore and suggest optimal hyperparameter configurations tailored to your needs.

## Installation

Currently, you need to install the tool manually. I will release it on PyPI soon.

## Basic Usage

Here is a basic example of how to use **cpsat-autotune** to optimize the time required to find an optimal solution for a CP-SAT model:

```python
from pathlib import Path
from cpsat_autotune import (
    tune_for_quality_within_timelimit,
    import_model,
    tune_time_to_optimal,
)

# Load your model
model = import_model(Path("models/medium_hg.pb"))

# Tune the model to minimize the time to reach an optimal solution
tune_time_to_optimal(
    model,
    timelimit_in_s=6,
    n_samples_per_param=5,
    max_samples_per_param=10,
    n_trials=50,
)
```

Sample output:
```plaintext
------------------------------------------------------------
Parameters:
	cp_model_probing_level: 1
	use_lb_relax_lns: True
	max_presolve_iterations: 1
	symmetry_level: 0
	ignore_subsolvers: ('fixed', 'no_lp', 'probing', 'quick_restart_no_lp')
	linearization_level: 2
	search_branching: 1
	randomize_search: True
	add_objective_cut: True
	diversify_lns_params: True
	preferred_variable_order: 1
	repair_hint: True
	max_all_diff_cut_size: 32
	use_erwa_heuristic: True
Difference to default: 3.2086112
This is significant: True
------------------------------------------------------------
------------------------------------------------------------
Parameters:
	presolve_probing_deterministic_time_limit: 10.0
	randomize_search: True
	add_objective_cut: True
	preferred_variable_order: 2
	diversify_lns_params: True
	max_all_diff_cut_size: 128
	symmetry_level: 0
Difference to default: 0.39306280000000005
This is significant: True
------------------------------------------------------------
------------------------------------------------------------
Parameters:
	presolve_bve_threshold: 100
	presolve_probing_deterministic_time_limit: 0.1
	search_branching: 1
	linearization_level: 2
	randomize_search: True
	ignore_subsolvers: ('default_lp', 'probing', 'quick_restart', 'quick_restart_no_lp')
	fp_rounding: 1
	max_all_diff_cut_size: 128
	symmetry_level: 0
Difference to default: 0.5854168
This is significant: True
------------------------------------------------------------
------------------------------------------------------------
Parameters:
	cp_model_probing_level: 1
	search_branching: 5
	presolve_bve_threshold: 100
	use_lb_relax_lns: True
	cut_level: 0
	use_objective_lb_search: True
	max_presolve_iterations: 2
	ignore_subsolvers: ('default_lp', 'max_lp', 'quick_restart', 'quick_restart_no_lp')
	diversify_lns_params: True
	use_erwa_heuristic: True
	presolve_probing_deterministic_time_limit: 10.0
Difference to default: 1.3500721999999996
This is significant: True
------------------------------------------------------------
------------------------------------------------------------
Parameters:
	cp_model_probing_level: 1
	use_lb_relax_lns: True
	symmetry_level: 0
	search_branching: 1
	presolve_probing_deterministic_time_limit: 5.0
	add_objective_cut: True
	diversify_lns_params: True
	use_erwa_heuristic: True
	max_presolve_iterations: 1
	repair_hint: True
	ignore_subsolvers: ('fixed', 'no_lp', 'probing', 'pseudo_costs', 'quick_restart_no_lp')
	cut_level: 0
Difference to default: 3.0537128
This is significant: True
------------------------------------------------------------
```

## Available Tuning Methods

**cpsat-autotune** provides two primary methods for tuning:

### 1. `tune_time_to_optimal`

This method tunes the CP-SAT solver's hyperparameters to minimize the time required to find an optimal solution. This is particularly useful for scenarios where time efficiency is critical.

#### Parameters:
- `model`: The CP-SAT model you wish to tune.
- `timelimit_in_s`: Time limit for each solver operation in seconds.
- `opt_gap`: (Optional) The relative optimality gap to determine when a solution is considered optimal. Default is `0.0` (exact optimality).
- `n_samples_per_param`: (Optional) Number of samples per parameter in each trial. Default is `10`.
- `max_samples_per_param`: (Optional) Maximum number of samples per parameter before using the mean to improve runtime. Default is `30`.
- `n_trials`: (Optional) Number of trials to run. Default is `100`.

### 2. `tune_for_quality_within_timelimit`

This method tunes hyperparameters to maximize or minimize the solution quality within a specified time limit.

#### Parameters:
- `model`: The CP-SAT model to be tuned.
- `timelimit_in_s`: Time limit for each solver operation in seconds.
- `obj_for_timeout`: Objective value applied if the solver times out.
- `direction`: Specify 'maximize' or 'minimize' depending on whether you want to optimize for the best or worst solution quality.
- `n_samples_per_param`: (Optional) Number of samples per parameter in each trial. Default is `10`.
- `max_samples_per_param`: (Optional) Maximum number of samples per parameter before using the mean to improve runtime. Default is `30`.
- `n_trials`: (Optional) Number of trials to run. Default is `100`.

## The Importance of Avoiding Overfitting

While tuning hyperparameters can improve solver performance for specific instances, it also increases the risk of overfitting. Overfitting occurs when the solver's performance is significantly improved on the training set of problems but deteriorates on new, slightly different instances. For example, tuning may reduce solve times on a set of similar problems but could result in excessive solve times or failure on problems that deviate from the training set.

### Recommendations:
- **Robust Performance:** If consistent performance across a variety of instances is crucial, stick with the default CP-SAT parameters.
- **Targeted Performance:** If you are solving a large number of similar problems and can tolerate potential performance drops on outliers, use the suggested parameters after careful consideration.

## Contributing

Contributions are welcome. Please ensure that your code adheres to the project's style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License.
