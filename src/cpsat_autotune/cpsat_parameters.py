"""
This file specifies the parameters that this tool can tune. Check out the original
file from OR-Tools for more information:
https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto
"""

from .parameters import (
    CpSatParameter,
    BoolParameter,
    CategoryParameter,
    IntParameter,
    ListParameter,
    IntFromOrderedListParameter,
)

CPSAT_PARAMETERS = [
    # ===============================================================
    # Branching and polarity
    # ===============================================================
    IntFromOrderedListParameter(
        name="preferred_variable_order",
        default_index=0,
        values=[0, 1, 2],
        description="""
Sets the initial order in which variables are selected during the search. The options are:

- `0` (IN_ORDER): Select variables in the order they appear in the problem definition.
- `1` (IN_REVERSE_ORDER): Select variables in reverse order.
- `2` (IN_RANDOM_ORDER): Select variables randomly.

The order of variable selection can have a significant impact on the efficiency of the search,
though, this is only the initial order and the solver will actually try to learn a better order.
The search will restart frequently such that this initial order should only influence the beginning
of the search.
        """,
    ),
    BoolParameter(
        name="use_erwa_heuristic",
        default_value=False,
        description="""
Enables the Exponential Recency Weighted Average (ERWA) heuristic for making branching decisions,
as described in "Learning Rate Based Branching Heuristic for SAT Solvers" (Liang et al., SAT 2016).
It considers the variable selection as an optimization problem that tries to maximize the learning rate,
i.e., the number of learned clauses per decision.
        """,
    ),
    BoolParameter(
        name="also_bump_variables_in_conflict_reasons",
        default_value=False,
        description="""
When enabled, the solver also bumps variables that are part of the conflict reason during conflict analysis.
This can help the solver avoid similar conflicts in the future by prioritizing these variables in the search.
        """,
    ),
    # ===============================================================
    # Conflict analysis
    # ===============================================================
    CategoryParameter(
        name="binary_minimization_algorithm",
        default_value=1,
        values=[0, 1, 2, 3, 4],
        description="""
Specifies the algorithm used for binary clause minimization during conflict analysis. The options are:

- `0` NO_BINARY_MINIMIZATION.
- `1` BINARY_MINIMIZATION_FIRST
- `2` BINARY_MINIMIZATION_WITH_REACHABILITY
- `3` EXPERIMENTAL_BINARY_MINIMIZATION
- `4` BINARY_MINIMIZATION_FIRST_WITH_TRANSITIVE_REDUCTION
        """,
    ),
    # ===============================================================
    # Clause database management
    # ===============================================================
    CategoryParameter(
        name="clause_cleanup_protection",
        default_value=0,
        values=[0, 1, 2],
        description="""
Specifies the level of protection against clause cleanup. The options are:

- `0` PROTECTION_NONE
- `1` PROTECTION_ALWAYS
- `2` PROTECTION_LBD
        """,
    ),
    # ===============================================================
    # Presolve
    # ===============================================================
    IntFromOrderedListParameter(
        name="presolve_bve_threshold",
        default_index=1,
        values=[100, 500, 1000],
        description="""
Determines the threshold for Bounded Variable Elimination (BVE) during presolve.
BVE eliminates variables that can be easily solved based on their limited impact. Lower thresholds speed up presolve but may result in less thorough simplification of the problem.
        """,
    ),
    IntFromOrderedListParameter(
        name="max_presolve_iterations",
        default_index=2,
        values=[1, 2, 3, 5, 10],
        description="""
Sets the maximum number of iterations that the presolve phase will execute.
Presolve simplifies the problem by eliminating redundant constraints and variables before the main search begins.
More iterations can lead to a more simplified problem but at the cost of longer presolve times.
        """,
    ),
    IntFromOrderedListParameter(
        name="cp_model_probing_level",
        default_index=2,
        values=[0, 1, 2],
        description="""
Defines the intensity of probing during presolve, where variables are temporarily fixed to infer more information about the problem.
Higher levels of probing can result in a more simplified problem but require more computation time during presolve.
        """,
    ),
    IntFromOrderedListParameter(
        name="presolve_probing_deterministic_time_limit",
        default_index=4,
        values=[0.1, 1.0, 5.0, 10.0, 30.0],
        description="""
Sets a deterministic time limit for probing during presolve.
This parameter ensures that the presolve phase does not consume too much time, allowing the solver to proceed to the main search phase in a timely manner.
        """,
    ),
    BoolParameter(
        "encode_complex_linear_constraint_with_integer",
        default_value=False,
        description="""
Introduces a slack variable with a domain equal to the right hand side for complex linear constraints.
https://github.com/google/or-tools/blob/2c333f58a37d7c75d29a58fd772c9b3f94e2ca1c/ortools/sat/cp_model_expand.cc#L1872
        """,
    ),
    # ===============================================================
    # Multithread
    # ===============================================================
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
Specifies a list of subsolvers to exclude from use during the search.
Removing certain subsolvers can free up resources and potentially speed up the search. However, excluding subsolvers can be risky as it might eliminate strategies that are useful for solving difficult instances.

From the OR-Tools documentation:

- `default_lp`           (linearization_level:1)
- `fixed`                (only if fixed search specified or scheduling)
- `no_lp`                (linearization_level:0)
- `max_lp`               (linearization_level:2)
- `pseudo_costs`         (only if objective, change search heuristic)
- `reduced_costs`        (only if objective, change search heuristic)
- `quick_restart`        (kind of probing)
- `quick_restart_no_lp`  (kind of probing with linearization_level:0)
- `lb_tree_search`       (to improve lower bound, MIP like tree search)
- `probing`              (continuous probing and shaving)
        """,
    ),
    # ===============================================================
    # Constraint programming parameters
    # ===============================================================
    CategoryParameter(
        name="search_branching",
        default_value=0,
        values=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        description="""
Defines the branching strategy the solver uses to navigate the search tree. The options are:

- `0` (AUTOMATIC_SEARCH): The solver automatically selects the most appropriate strategy.
- `1` (FIXED_SEARCH): Follows a fixed variable order, as specified by the user or the problem model.
- `2` (PORTFOLIO_SEARCH): Uses a combination of multiple strategies to explore the search space.
- `3` (LP_SEARCH): Branches based on the LP relaxation of the problem, leveraging the reduced costs of variables.
- `4` (PSEUDO_COST_SEARCH): Branches using pseudo-costs, which are estimates of the impact of branching decisions based on past experiences.
- `5` (PORTFOLIO_WITH_QUICK_RESTART_SEARCH): Quickly explores different heuristics with low conflict limits, aiming to find a good initial solution.
- `6` (HINT_SEARCH): Prioritizes decisions based on hints provided by the user or the problem model.
- `7` (PARTIAL_FIXED_SEARCH): Begins with a fixed strategy, then switches to automatic search for the remaining decisions.
- `8` (RANDOMIZED_SEARCH): Introduces randomization into branching decisions to diversify the search.
        """,
    ),
    BoolParameter(
        name="repair_hint",
        default_value=False,
        description="""
Enables the solver to attempt to repair a solution based on a provided hint before switching to a general search strategy.
This can be useful when a good initial guess is available, helping the solver find a feasible solution more quickly.
        """,
    ),
    BoolParameter(
        name="use_lns_only",
        default_value=False,
        description="""
When enabled, the solver uses only Large Neighborhood Search (LNS) heuristics, without performing a full global search.
LNS is beneficial in scenarios where finding improvements quickly is more important than exploring all possibilities for an optimal solution.
        """,
    ),
    BoolParameter(
        name="use_lb_relax_lns",
        default_value=False,
        description="""
Activates a neighborhood generation approach based on local branching combined with linear programming (LP) relaxation,
as described in "Local Branching Relaxation Heuristics for Integer Linear Programs" (Huang et al., 2023).
This method can help the solver explore the solution space more effectively by focusing on promising regions.
        """,
    ),
    BoolParameter(
        name="use_objective_lb_search",
        default_value=False,
        description="""
Guides the solver to start its search by focusing on improving the lower bound of the objective value.
This approach can help direct the solver toward the most promising regions of the solution space, especially when minimizing the objective.
        """,
    ),
    BoolParameter(
        name="use_objective_shaving_search",
        default_value=False,
        description="""
Activates a search strategy that aggressively restricts the objective value range, effectively reducing the search space.
This is useful in problems with tight constraints, where such focused searches can lead to faster identification of optimal solutions.
        """,
    ),
    BoolParameter(
        name="optimize_with_core",
        default_value=False,
        description="""
Use a core-based approach when trying to improve the bound.
        """,
    ),
    IntParameter(
        name="feasibility_jump_linearization_level",
        default_value=2,
        lb=0,
        ub=2,
        log=False,
        description="""
        Linearization level for feasibility jump.
        """,
    ),
    CategoryParameter(
        name="fp_rounding",
        default_value=2,
        values=[0, 1, 3, 2],
        description="""
Specifies the rounding method used in the feasibility pump, a heuristic for quickly finding integer feasible solutions. The options are:

- `0` (NEAREST_INTEGER): Rounds values to the nearest integer.
- `1` (LOCK_BASED): Rounds based on the direction with fewer constraints ("locks").
- `2` (PROPAGATION_ASSISTED): Rounds with consideration of bound propagation, improving solution quality by integrating more logical deductions.
- `3` (ACTIVE_LOCK_BASED): Similar to lock-based rounding, but focuses on active constraints from the last LP solve.
        """,
    ),
    BoolParameter(
        name="diversify_lns_params",
        default_value=False,
        description="""
Enables the use of varied parameter settings for Large Neighborhood Search (LNS).
By diversifying these parameters, the solver can explore different areas of the solution space more effectively, increasing the likelihood of finding better solutions.
        """,
    ),
    BoolParameter(
        name="polish_lp_solution",
        default_value=False,
        description="""
Activates a polishing step that refines the solution found by the LP solver. Expensive but can help for some problems.
        """,
    ),
    # ===============================================================
    # Linear programming relaxation
    # ===============================================================
    IntFromOrderedListParameter(
        name="linearization_level",
        default_index=1,
        values=[0, 1, 2],
        description="""
Controls the extent to which integer constraints are transformed into Boolean variables for LP relaxation. The levels are:

- `0`: No linearization, the solver does not use LP relaxation.
- `1`: Basic linearization, including linear constraints and fully encoded Boolean variables.
- `2`: More comprehensive linearization, including Boolean constraints.

Increasing the linearization level can tighten the relaxation, but it also increases the complexity of the model.
        """,
    ),
    BoolParameter(
        name="add_objective_cut",
        default_value=False,
        description="""
Controls whether to add cuts based on the fractional objective value to the model.
These cuts, when enabled, help narrow down the feasible region of the problem, potentially speeding up convergence to an optimal solution by eliminating non-promising areas.
        """,
    ),
    IntParameter(
        name="cut_level",
        default_value=1,
        lb=0,
        ub=2,
        log=False,
        description="""
Sets the level of effort the solver will invest in generating cutting planes, which are linear constraints added to remove infeasible regions.
Properly applied, cuts can significantly reduce the search space and help the solver find an optimal solution more quickly.
        """,
    ),
    IntFromOrderedListParameter(
        name="max_all_diff_cut_size",
        default_index=1,
        values=[32, 64, 128],
        description="""
Limits the size of "all different" constraints used when generating cuts.
All-different constraints ensure that a set of variables takes distinct values. This parameter controls the balance between reducing the search space and the computational cost of generating cuts.
        """,
    ),
    IntFromOrderedListParameter(
        name="symmetry_level",
        default_index=2,
        values=[0, 1, 2],
        description="""
Specifies the level of symmetry detection and exploitation during the search.
Symmetry breaking helps reduce redundant work by avoiding the exploration of equivalent solutions, thus speeding up the search.

- `0`: No symmetry breaking.
- `1`: Detect and break symmetries in presolve.
- `2`: Use both presolve and dynamic symmetry breaking during the search.
        """,
    ),
]


def get_parameter_by_name(name: str) -> CpSatParameter:
    """
    Returns the parameter with the given name.
    """
    for param in CPSAT_PARAMETERS:
        if param.name == name:
            return param
    raise KeyError(f"Parameter '{name}' not found.")
