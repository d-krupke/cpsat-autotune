# cpsat-autotune: A Tool for Tuning the Hyper-parameters of Google OR-Tools' CP-SAT Solver

This project aims to provide a simple tool that suggests how to tune the
hyper-parameters of the CP-SAT solver for a specific problem. While CP-SAT is
already optimized for generic problems, tuning the solver for a specific problem
can potentially improve performance, especially when dealing with a large number
of similar problems. However, it is important to note that significant
improvements should not be expected in all cases.

```python
tune_for_quality_within_timelimit(
    models[0], timelimit_in_s=1, obj_for_timeout=0, direction="maximize"
)
```

```output
The best parameters are: {'symmetry_level': 0, 'presolve_probing_deterministic_time_limit': 1.0, 'preferred_variable_order': 1, 'randomize_search': True, 'max_presolve_iterations': 1, 'fp_rounding': 0, 'cut_level': 0, 'cp_model_probing_level': 0, 'linearization_level': 0, 'search_branching': 6}
The difference to the baseline is: 431.6000000000058
This is significant: True
```

> :warning: This is an early prototype.

Internally, this tool utilizes the `optuna` library to optimize the
hyper-parameters of the CP-SAT solver.

## The Importance of Avoiding Overfitting

When using this tool, it is crucial to be aware of the risks associated with
overfitting to specific problem instances. While the tuned CP-SAT solver may
exhibit improved performance on your test instances, it is possible that its
performance may significantly deteriorate on even slightly different instances.
For example, if you have a sequence of 5 problems, each taking an average of 5
seconds to solve with the default parameters, tuning the solver may reduce the
solving time for the first four problems to 2 seconds. However, the last
problem, which deviates slightly from the others, may now take 10 minutes to
solve or may not be solvable at all due to the deactivation of a strategy that
was not relevant for the first four problems but crucial for the last.
Therefore, if you require robust performance across all instances, it is
recommended to stick with the well-tuned default parameters. However, if you can
tolerate potential performance drops on a few instances and are willing to
carefully consider the suggested parameters instead of blindly applying them,
this tool may prove useful.
