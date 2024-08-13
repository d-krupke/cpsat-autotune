
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
from google.protobuf import text_format
import optuna
from .parameters import BoolParameter, CategoryParameter, IntParameter, ListParameter, IntFromOrderedListParameter


class CpSatParameterSpace:
    """
    Defines the hyperparameter space for the CP-SAT solver to be optimized by Optuna.

    The parameters are currently not too well chosen, as this is still the prototype.
    Based on empirical testing, the selection should be refined.
    """
    parameter_space = [
        BoolParameter(name="use_lns_only", default_value=False),
        BoolParameter(name="repair_hint", default_value=False),
        BoolParameter(name="use_lb_relax_lns", default_value=False),
        IntFromOrderedListParameter(name="preferred_variable_order", default_index=0, values=[0, 1, 2]),
        BoolParameter(name="use_erwa_heuristic", default_value=False),
        IntFromOrderedListParameter(name="linearization_level", default_index=1, values=[0, 1, 2]),
        CategoryParameter(name="fp_rounding", default_value=3, values=[0, 1, 3, 2]),
        BoolParameter(name="randomize_search", default_value=False),
        BoolParameter(name="diversify_lns_params", default_value=False),
        BoolParameter(name="add_objective_cut", default_value=False),
        BoolParameter(name="use_objective_lb_search", default_value=False),
        BoolParameter(name="use_objective_shaving_search", default_value=False),
        CategoryParameter(name="search_branching", default_value=0, values=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        IntFromOrderedListParameter(name="cut_level", default_index=1, values=[0, 1]),
        IntFromOrderedListParameter(name="max_all_diff_cut_size", default_index=1, values=[32, 64, 128]),
        IntFromOrderedListParameter(name="symmetry_level", default_index=2, values=[0, 1, 2]),
        # Controls the maximum number of iterations the presolve phase will execute. Fewer iterations can lead to faster presolve times but may result in less simplification of the problem.
        IntFromOrderedListParameter(name="max_presolve_iterations", default_index=2, values=[1, 2, 3, 5, 10]),
        # Sets the level of effort for probing during presolve. Lower levels reduce the time spent on probing but may leave more complex structures in the problem.
        IntFromOrderedListParameter(name="cp_model_probing_level", default_index=2, values=[0, 1, 2]),
        # Limits the deterministic time allocated for probing during presolve. Adjusting this can prevent excessive time consumption in the presolve phase.
        IntFromOrderedListParameter(name="presolve_probing_deterministic_time_limit", default_index=4, values=[0.1, 1.0, 5.0, 10.0, 30.0]),
        # Sets the threshold for bounded variable elimination (BVE) during presolve. Lower thresholds can speed up presolve but might result in less effective elimination of redundant variables.
        IntFromOrderedListParameter(name="presolve_bve_threshold", default_index=1, values=[100, 500, 1000]),
        ListParameter(name="ignore_subsolvers", default_value = [], values = ["default_lp", "fixed", "no_lp", "max_lp", "pseudo_costs", "reduced_costs", "quick_restart", "quick_restart_no_lp", "lb_tree_search", "probing"])
    ]
    def __init__(self, max_difference_to_default: int = -1):
        self.fixed_parameters = {}
        self.tunable_parameters = {param.name: param for param in self.parameter_space}
        self.max_difference_to_default = max_difference_to_default

    def fix_parameter(self, parameter: str, value = None):
        """
        Will fix a parameter. If the value is None, the default of CP-SAT will be used.
        The value must be for CP-SAT, not for Optuna.
        """
        self.tunable_parameters.pop(parameter, None)
        if value is not None:
            self.fixed_parameters[parameter] = value

    def sample(self, trial: optuna.Trial|optuna.trial.FixedTrial|dict|None, solver: cp_model.CpSolver|None = None) -> cp_model.CpSolver:
        """
        Returns a configured CP-SAT solver for a trial by Optuna. If no trial is given, the default solver is returned.
        """
        solver = cp_model.CpSolver() if solver is None else solver
        for parameter, value in self.fixed_parameters.items():
            setattr(solver.parameters, parameter, value)
        if trial is None:
            return solver
        num_different = 0
        if isinstance(trial, dict):
            trial = optuna.trial.FixedTrial(trial)
        for parameter in self.tunable_parameters.values():
            if parameter.name not in self.fixed_parameters:
                value = parameter.sample(trial)
                if value != parameter.get_cpsat_default():
                    num_different += 1
                    if self.max_difference_to_default >= 0 and num_different > self.max_difference_to_default:
                        raise optuna.TrialPruned()
                value = parameter.sample(trial)
                if isinstance(value, list):
                    param_handle = getattr(solver.parameters, parameter.name).extend(value)
                else:
                    setattr(solver.parameters, parameter.name, parameter.sample(trial))
        return solver
    
    def get_default_params_for_optuna(self):
        """
        Returns the default parameters for Optuna. These values can be different to the values used by CP-SAT.
        """
        default_params = {}
        for param in self.tunable_parameters.values():
            default_params.update(param.get_optuna_default())
        return default_params
    
    def build_solver_for_params(self, cpsat_params: dict):
        solver = cp_model.CpSolver()
        for parameter, value in cpsat_params.items():
            setattr(solver.parameters, parameter, value)
        return solver
    
    def get_cpsat_params_from_trial(self, trial: optuna.trial.FixedTrial) -> dict:
        """
        Returns the parameters for CP-SAT from an Optuna trial. The values are not the Optuna values, but the CP-SAT values.
        """
        return {param_name: self.tunable_parameters[param_name].get_cpsat_param(optuna_param_value) for param_name, optuna_param_value in trial.params.items()}
    
    def get_cpsat_params_diff(self, trial: optuna.trial.FixedTrial|optuna.Trial) -> dict:
        """
        Returns the parameters that are different from the default.
        """
        params = {}
        for param in self.tunable_parameters.values():
            params.update(param.get_cpsat_params(trial.params))
        cp_sat_default = frozenset((param.name, param.get_cpsat_default()) for param in self.tunable_parameters.values())
        trial_params = frozenset(params.items())
        diff_params = trial_params - cp_sat_default
        return {param_name: param_value for param_name, param_value in diff_params}
        default_params = self.get_
        return {param_name: self.tunable_parameters[param_name].get_cpsat_param(optuna_param_value) for param_name, optuna_param_value in trial.params.items() if optuna_param_value != self.tunable_parameters[param_name].get_optuna_default()}
    
    def distance_to_default(self, trial: optuna.trial.FixedTrial) -> int:
        """
        Returns the number of parameters that are different from the default.
        This can be useful as we usually want to use as many default values as possible,
        as they are much better tested.
        """
        return len(self.get_cpsat_params_diff(trial))