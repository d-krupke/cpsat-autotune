from abc import ABC, abstractmethod
import optuna


class CpSatParameter(ABC):
    def __init__(self, name, default_value):
        self.name = name
        self._default_value = default_value

    @abstractmethod
    def sample(self, trial: optuna.Trial):
        pass

    def get_optuna_default(self) -> dict:
        return {self.name: self._default_value}

    def get_cpsat_default(self):
        return self._default_value

    def get_cpsat_params(self, optuna_params):
        return {self.name: optuna_params[self.name]}

    def get_optuna_params(self, cpsat_params):
        return {self.name: cpsat_params[self.name]}


class BoolParameter(CpSatParameter):
    def __init__(self, name, default_value):
        super().__init__(name, default_value)

    def sample(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, [True, False])


class CategoryParameter(CpSatParameter):
    def __init__(self, name, default_value, values):
        super().__init__(name, default_value)
        self.values = values

    def sample(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, self.values)


class IntParameter(CpSatParameter):
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
