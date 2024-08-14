from abc import ABC, abstractmethod
import optuna


class CpSatParameter(ABC):
    """
    An abstract class for a parameter of CP-SAT that can be optimized with Optuna.
    Optuna and CP-SAT may need different representations of the parameters, so this class
    provides methods to convert between the two.
    """
    def __init__(self, name, default_value):
        self.name = name
        self._default_value = default_value

    @abstractmethod
    def sample(self, trial: optuna.Trial):
        pass

    def get_optuna_default(self) -> dict:
        """
        Return the default value in the format that Optuna expects.
        As for example lists need to be split into multiple parameters, this method
        returns a dictionary.
        """
        return {self.name: self._default_value}

    def get_cpsat_default(self):
        """
        Return the default value in the format that CP-SAT expects.
        As this matches a single CP-SAT parameter, this method returns the value itself,
        differently from get_optuna_default.
        """
        return self._default_value

    def get_cpsat_params(self, optuna_params: dict) -> dict:
        """
        Convert the Optuna parameters to CP-SAT parameters.
        """
        return {self.name: optuna_params[self.name]}

    def get_optuna_params(self, cpsat_params: dict) -> dict:
        """
        Convert the CP-SAT parameters to Optuna parameters.
        """
        return {self.name: cpsat_params[self.name]}


class BoolParameter(CpSatParameter):
    """
    A simple True/False parameter of CP-SAT.
    """
    def __init__(self, name, default_value):
        super().__init__(name, default_value)

    def sample(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, [True, False])


class CategoryParameter(CpSatParameter):
    """
    A parameter of CP-SAT that can take a value from a list of values.
    The order of the values does not have any semantic meaning.
    If the order has a meaning, use IntFromOrderedListParameter instead, as it allows Optuna to make assumptions about the order.
    """
    def __init__(self, name, default_value, values):
        super().__init__(name, default_value)
        self.values = values

    def sample(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, self.values)


class IntParameter(CpSatParameter):
    """
    A parameter of CP-SAT that is an integer.
    """
    def __init__(self, name, default_value, lb, ub, log: bool):
        super().__init__(name, default_value)
        self.lower_bound = lb
        self.upper_bound = ub
        self.log = log

    def sample(self, trial: optuna.Trial):
        return trial.suggest_int(
            self.name, low=self.lower_bound, high=self.upper_bound, log=self.log
        )


class ListParameter(CpSatParameter):
    """
    A parameter of CP-SAT that is a list of values and we need to select a subset of them.
    This is a complex parameter that is not directly supported by Optuna, thus, we need to split it into multiple parameters.
    """
    def __init__(self, name, default_value, values):
        super().__init__(name, tuple(sorted(default_value)))
        self.values = values

    def sample(self, trial: optuna.Trial):
        sampled_list = []
        for value in self.values:
            select = trial.suggest_categorical(f"{self.name}:{value}", [True, False])
            if select:
                sampled_list.append(value)
        return sampled_list

    def get_optuna_default(self) -> dict:
        return {
            f"{self.name}:{value}": value in self._default_value
            for value in self.values
        }

    def get_cpsat_params(self, optuna_params):
        return {
            self.name: tuple(
                sorted(
                    value
                    for value in self.values
                    if optuna_params[f"{self.name}:{value}"]
                )
            )
        }

    def get_optuna_params(self, cpsat_params):
        return {
            f"{self.name}:{value}": value in cpsat_params[self.name]
            for value in self.values
        }


class IntFromOrderedListParameter(CpSatParameter):
    """
    A mixture of categorical and integer parameter. The parameter has to be from
    a list of values, but the order has a semantic meaning, meaning that entry 1 is
    semantically between 0 and 2, etc. Separating this from a categorical parameter
    allows Optuna to make assumptions about the order of the values.

    CAVEAT: The default value has to be the index of the value in the list, not the value itself.
    """

    def __init__(self, name, default_index, values):
        super().__init__(name, default_index)
        self.values = values

    def sample(self, trial: optuna.Trial):
        return self.values[
            trial.suggest_int(self.name, low=0, high=len(self.values) - 1)
        ]

    def get_optuna_default(self):
        return {self.name: self._default_value}

    def get_cpsat_default(self):
        return self.values[self._default_value]

    def get_cpsat_params(self, optuna_params):
        return {self.name: self.values[optuna_params[self.name]]}

    def get_optuna_params(self, cpsat_params):
        value = cpsat_params[self.name]
        return {self.name: self.values.index(value)}
