import logging
from typing import Iterable
from ortools.sat.python import cp_model
import optuna
from .cpsat_parameters import CPSAT_PARAMETERS, CpSatParameter


class CpSatParameterSpace:
    """
    Defines the hyperparameter space for the CP-SAT solver to be optimized by Optuna.

    The parameters are currently not too well chosen, as this is still the prototype.
    Based on empirical testing, the selection should be refined.
    """

    def __init__(self):
        self.tunable_parameters = {param.name: param for param in CPSAT_PARAMETERS}

    def drop_parameter(self, parameter: str):
        """
        Will remove a parameter from the parameter space.
        """
        self.tunable_parameters.pop(parameter, None)

    def filter_applicable_parameters(self, models: Iterable[cp_model.CpModel]):
        """
        Filters the parameter space to only include parameters that are applicable to the given models.

        This method iterates over the tunable parameters and removes those that are not effective 
        for any of the provided models.

        Args:
            models (Iterable[cp_model.CpModel]): An iterable of CP-SAT models to check parameter applicability against.
        """
        params = list(self.tunable_parameters.values())
        for param in params:
            param: CpSatParameter
            if not any(param.is_effective_for(model) for model in models):
                logging.info("Dropping parameter `%s` as it is not effective for any of the provided models.", param.name)
                self.drop_parameter(param.name)

    def sample(
        self,
        trial: optuna.Trial | optuna.trial.FixedTrial | dict | None,
    ) -> dict[str, float | int | bool | list | tuple]:
        """
        Returns a sample of the parameters for CP-SAT.
        """
        if trial is None:
            return {}
        if isinstance(trial, dict):
            trial = optuna.trial.FixedTrial(trial)
        assert isinstance(trial, (optuna.Trial, optuna.trial.FixedTrial))
        params = {}
        for parameter in self.tunable_parameters.values():
            value = parameter.sample(trial)
            default = parameter.get_cpsat_default()
            if isinstance(value, (list, tuple)):
                if set(default) != set(value):
                    params[parameter.name] = list(value)
            elif value != default:
                params[parameter.name] = value
        return params

    def get_default_params_for_optuna(self):
        """
        Returns the default parameters for Optuna. These values can be different to the values used by CP-SAT.
        """
        default_params = {}
        for param in self.tunable_parameters.values():
            default_params.update(param.get_optuna_default())
        return default_params

    def get_cpsat_params_from_trial(self, trial: optuna.trial.FixedTrial) -> dict:
        """
        Returns the parameters for CP-SAT from an Optuna trial. The values are not the Optuna values, but the CP-SAT values.
        """
        return {
            param_name: self.tunable_parameters[param_name].get_cpsat_param(
                optuna_param_value
            )
            for param_name, optuna_param_value in trial.params.items()
        }
