from .parameters import (
    BoolParameter,
    CategoryParameter,
    ListParameter,
    IntFromOrderedListParameter,
)

CPSAT_PARAMETERS = [
    BoolParameter(
        name="use_lns_only",
        default_value=False,
        description="""
        Use only heuristics (especially LNS) to solve the problem instead of a full search.
        In some cases, this can lead to faster improvements but it is not as powerful as a full search.
        """,
    ),
    BoolParameter(name="repair_hint", default_value=False),
    BoolParameter(name="use_lb_relax_lns", default_value=False),
    IntFromOrderedListParameter(
        name="preferred_variable_order", default_index=0, values=[0, 1, 2]
    ),
    BoolParameter(name="use_erwa_heuristic", default_value=False),
    IntFromOrderedListParameter(
        name="linearization_level", default_index=1, values=[0, 1, 2]
    ),
    CategoryParameter(name="fp_rounding", default_value=3, values=[0, 1, 3, 2]),
    BoolParameter(name="randomize_search", default_value=False),
    BoolParameter(name="diversify_lns_params", default_value=False),
    BoolParameter(name="add_objective_cut", default_value=False),
    BoolParameter(name="use_objective_lb_search", default_value=False),
    BoolParameter(name="use_objective_shaving_search", default_value=False),
    CategoryParameter(
        name="search_branching", default_value=0, values=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ),
    IntFromOrderedListParameter(name="cut_level", default_index=1, values=[0, 1]),
    IntFromOrderedListParameter(
        name="max_all_diff_cut_size", default_index=1, values=[32, 64, 128]
    ),
    IntFromOrderedListParameter(
        name="symmetry_level", default_index=2, values=[0, 1, 2]
    ),
    # Controls the maximum number of iterations the presolve phase will execute. Fewer iterations can lead to faster presolve times but may result in less simplification of the problem.
    IntFromOrderedListParameter(
        name="max_presolve_iterations", default_index=2, values=[1, 2, 3, 5, 10]
    ),
    # Sets the level of effort for probing during presolve. Lower levels reduce the time spent on probing but may leave more complex structures in the problem.
    IntFromOrderedListParameter(
        name="cp_model_probing_level", default_index=2, values=[0, 1, 2]
    ),
    # Limits the deterministic time allocated for probing during presolve. Adjusting this can prevent excessive time consumption in the presolve phase.
    IntFromOrderedListParameter(
        name="presolve_probing_deterministic_time_limit",
        default_index=4,
        values=[0.1, 1.0, 5.0, 10.0, 30.0],
    ),
    IntFromOrderedListParameter(
        name="presolve_bve_threshold",
        default_index=1,
        values=[100, 500, 1000],
        description="""
        Sets the threshold for bounded variable elimination (BVE) during presolve.
        Lower thresholds can speed up presolve but might result in less effective
        elimination of redundant variables.
        """,
    ),
    ListParameter(
        name="ignore_subsolvers",
        default_value=[],
        values=[
            "default_lp",
            "fixed",
            "no_lp",
            "max_lp",
            "pseudo_costs",
            "reduced_costs",
            "quick_restart",
            "quick_restart_no_lp",
            "lb_tree_search",
            "probing",
        ],
        description="""
        Subsolvers not to use. This can free up resources and speed up the search.
        However, it is also risky as you might remove a subsolver that is, e.g., only
        useful for difficult instances, but required for those.
        """,
    ),
]
