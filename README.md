# cpsat-autotune: A Hyperparameter Tuning Tool for Google's OR-Tools CP-SAT Solver

**cpsat-autotune** is a Python library designed to optimize the hyperparameters
of Google's OR-Tools CP-SAT solver for specific problem instances. While CP-SAT
is already highly optimized for a broad range of generic problems, fine-tuning
its parameters for particular problem sets can yield significant performance
gains. This tool leverages the `optuna` optimization library to systematically
explore and suggest optimal hyperparameter configurations tailored to your
needs.

Also check out our other projects:

- [The CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/): A comprehensive
  guide to Google's OR-Tools CP-SAT solver.
- [CP-SAT Log Analyzer](https://github.com/d-krupke/CP-SAT-Log-Analyzer): A tool
  to analyze the logs generated by Google's OR-Tools CP-SAT solver.

## Use Case

**cpsat-autotune** is not a universal solution that guarantees a performance
boost for all uses of the CP-SAT solver. Instead, it is specifically designed to
enhance solver efficiency in targeted scenarios, particularly within the context
of Adaptive Large Neighborhood Search (ALNS).

### Adaptive Large Neighborhood Search (ALNS) Context

In ALNS, the CP-SAT solver is frequently invoked with a strict time limit to
solve similar problem instances as part of a larger iterative optimization
process. The goal is to incrementally improve a solution by exploring different
neighborhoods of the problem space. In this context, achieving even modest
performance gains on average instances can significantly impact the overall
efficiency of the search process, even if it results in occasional performance
drops on outlier instances.

### Benefits of Tuning in ALNS

- **Average Performance Gains:** By tuning the solver’s hyperparameters to
  optimize performance on typical instances, **cpsat-autotune** can reduce the
  average time per iteration. This is particularly valuable in ALNS, where a
  large number of solver calls are made.
- **Tolerance for Outliers:** In an ALNS framework, occasional slower iterations
  due to deteriorated performance on outlier instances are generally acceptable,
  as the search process can recover in subsequent iterations. Thus, the focus
  can be on enhancing the solver's average performance rather than ensuring
  consistent performance across all instances.
- **Augmented Solver Strategies:** Instead of completely replacing the CP-SAT
  solver with a single tuned configuration, **cpsat-autotune** allows you to
  tune hyperparameters for one or more specific instance sets and incorporate
  these as additional strategies within ALNS. This means you can maintain the
  default CP-SAT parameters while augmenting the solver's capability with
  tailored configurations. ALNS can then automatically select the most effective
  strategy for each iteration, leveraging the diverse set of tuned
  hyperparameters alongside the default configuration for optimal performance.

## Installation

You can install **cpsat-autotune** using `pip`:

```bash
pip install -U cpsat-autotune
```

Make sure to update the package before every use to ensure you have the latest
version, as this project is still a prototype.

## Basic Usage

Here is a basic example of how to use **cpsat-autotune** to optimize the time
required to find an optimal solution for a CP-SAT model:

```python
from cpsat_autotune import import_model, tune_time_to_optimal

# Load your model from a protobuf file
model = import_model("models/medium_hg.pb")

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

This method tunes the CP-SAT solver's hyperparameters to minimize the time
required to find an optimal solution. This is useful if you need a guaranteed
solution without a fixed time limit.

#### Parameters:

- `model`: The CP-SAT model you wish to tune.
- `timelimit_in_s`: Time limit for each solver operation in seconds.
- `opt_gap`: (Optional) The relative optimality gap to determine when a solution
  is considered optimal. Default is `0.0` (exact optimality).
- `n_samples_per_param`: (Optional) Number of samples per parameter in each
  trial. Default is `10`.
- `max_samples_per_param`: (Optional) Maximum number of samples per parameter
  before using the mean to improve runtime. Default is `30`.
- `n_trials`: (Optional) Number of trials to run. Default is `100`.

### 2. `tune_for_quality_within_timelimit`

This method tunes hyperparameters to maximize or minimize the objective value
within a specified time limit. This is useful when you need to find a good
solution within a fixed time frame, but do not require any guarantees.

#### Parameters:

- `model`: The CP-SAT model to be tuned.
- `timelimit_in_s`: Time limit for each solver operation in seconds.
- `obj_for_timeout`: Objective value applied if the solver times out.
- `direction`: Specify 'maximize' or 'minimize' depending on whether you want to
  optimize for the best or worst solution quality.
- `n_samples_per_param`: (Optional) Number of samples per parameter in each
  trial. Default is `10`.
- `max_samples_per_param`: (Optional) Maximum number of samples per parameter
  before using the mean to improve runtime. Default is `30`.
- `n_trials`: (Optional) Number of trials to run. Default is `100`.

## The Importance of Avoiding Overfitting

While tuning hyperparameters can improve solver performance for specific
instances, it also increases the risk of overfitting. Overfitting occurs when
the solver's performance is significantly improved on the training set of
problems but deteriorates on new, slightly different instances. For example,
tuning may reduce solve times on a set of similar problems but could result in
excessive solve times or failure on problems that deviate from the training set.

## How does the Tuning Work?

**cpsat-autotune** uses the `optuna` library to perform hyperparameter tuning
on a preselected set of parameters. The output of optuna is then further refined
and the significance of certain parameters is evaluated. Based on the assumption
that the default parameters are already well-tuned for a broad range of problems,
**cpsat-autotune** identifies the most significant changes to the default
configuration and suggests these as potential improvements. It does take a few
shortcuts to speed things up, while collecting more samples for important
values.

### Recommendations:

- **Robust Performance:** If consistent performance across a variety of
  instances is crucial, stick with the default CP-SAT parameters.
- **Targeted Performance:** If you are solving a large number of similar
  problems and can tolerate potential performance drops on outliers, use the
  suggested parameters after careful consideration.

## Contributing

Contributions are welcome. Please ensure that your code adheres to the project's
style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License.
