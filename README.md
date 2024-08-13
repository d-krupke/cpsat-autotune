# autotune-CP-SAT: A tool to tune the hyper-parameters of Google OR-Tools' CP-SAT solver

This project aims to provide a simple tool to give a suggestion on how to tune
the hyper-parameters of the CP-SAT solver for a specific problem. CP-SAT is
already tuned for generic problems, but if you have to solve a lot of very
similar problems, you can gain some extra performance by tuning the solver for
your specific problem. You should not expect any significant improvement, but it
can be useful in some cases.

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

> :warning: This is a super early prototype.

Internally, it uses the `optuna` library to optimize the hyper-parameters of the
CP-SAT solver.
