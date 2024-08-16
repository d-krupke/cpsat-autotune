import optuna
from .cpsat_parameters import CPSAT_PARAMETERS


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
        params = {}
        for parameter in self.tunable_parameters.values():
            value = parameter.sample(trial)
            if value != parameter.get_cpsat_default():
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

    def get_cpsat_params_diff(
        self, trial: optuna.trial.FixedTrial | optuna.Trial
    ) -> dict:
        """
        Returns the parameters that are different from the default.
        """
        params = {}
        for param in self.tunable_parameters.values():
            params.update(param.get_cpsat_params(trial.params))
        cp_sat_default = frozenset(
            (param.name, param.get_cpsat_default())
            for param in self.tunable_parameters.values()
        )
        trial_params = frozenset(params.items())
        diff_params = trial_params - cp_sat_default
        return {param_name: param_value for param_name, param_value in diff_params}

    def distance_to_default(self, trial: optuna.trial.FixedTrial) -> int:
        """
        Returns the number of parameters that are different from the default.
        This can be useful as we usually want to use as many default values as possible,
        as they are much better tested.
        """
        return len(self.get_cpsat_params_diff(trial))
