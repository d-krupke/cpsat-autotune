"""
This class provides some auxiliary classes to define the parameters that can be optimized using Optuna.
This allows to map between the Optuna and CP-SAT parameter formats, making it easier to integrate the two frameworks.
It also provides more types of parameters than Optuna does out of the box, such as list parameters.
"""

from abc import ABC, abstractmethod
from ortools.sat.python import cp_model
from typing import Callable
import optuna


def _for_all_models(model: cp_model.CpModel) -> bool:
    """
    Default filter function that returns True for all models.
    """
    return True


class CpSatParameter(ABC):
    """
    Abstract base class representing a CP-SAT parameter that can be optimized using Optuna.
    This class defines the interface for converting parameters between CP-SAT and Optuna formats,
    allowing for seamless integration between the two frameworks.
    """

    def __init__(
        self,
        name: str,
        default_value,
        description: str = "",
        subsolver: bool = True,
        is_applicable_for: Callable[[cp_model.CpModel], bool] = _for_all_models,
    ):
        """
        Initialize the parameter with a name and default value.

        Args:
            name: The name of the parameter.
            default_value: The default value for the parameter.
        """
        self.name = name
        self._default_value = default_value
        self.description = description
        self._filter = is_applicable_for
        self.subsolver = subsolver

    @abstractmethod
    def sample(self, trial: optuna.Trial):
        """
        Abstract method to define how the parameter should be sampled during optimization.

        Args:
            trial: An Optuna trial object used to suggest parameter values.

        Returns:
            The sampled value for the parameter.
        """
        pass

    def get_optuna_default(self) -> dict:
        """
        Retrieve the default value of the parameter formatted for Optuna.

        Returns:
            A dictionary representing the default value in Optuna's expected format.
        """
        return {self.name: self._default_value}

    def get_cpsat_default(self):
        """
        Retrieve the default value of the parameter formatted for CP-SAT.

        Returns:
            The default value in CP-SAT's expected format.
        """
        return self._default_value

    def get_cpsat_params(self, optuna_params: dict) -> dict:
        """
        Convert Optuna parameter values to CP-SAT parameter values.

        Args:
            optuna_params: A dictionary of parameter values suggested by Optuna.

        Returns:
            A dictionary of parameter values formatted for CP-SAT.
        """
        return {self.name: optuna_params[self.name]}

    def get_optuna_params(self, cpsat_params: dict) -> dict:
        """
        Convert CP-SAT parameter values to Optuna parameter values.

        Args:
            cpsat_params: A dictionary of parameter values from CP-SAT.

        Returns:
            A dictionary of parameter values formatted for Optuna.
        """
        return {self.name: cpsat_params[self.name]}

    def is_effective_for(self, model: cp_model.CpModel) -> bool:
        """
        Returns true if the parameter could have an effect on solving the model.
        Will return false if it won't have any effect on solving the model.
        """
        return self._filter(model)

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return f"{self.name} [default: {self.get_cpsat_default()}]"


class BoolParameter(CpSatParameter):
    """
    A CP-SAT parameter representing a boolean (True/False) value.
    """

    def __init__(
        self,
        name: str,
        default_value: bool,
        description: str = "",
        subsolver: bool = True,
        is_applicable_for: Callable[[cp_model.CpModel], bool] = _for_all_models,
    ):
        """
        Initialize the boolean parameter with a name and default value.

        Args:
            name: The name of the parameter.
            default_value: The default boolean value for the parameter.
        """
        super().__init__(
            name,
            default_value=default_value,
            description=description,
            subsolver=subsolver,
            is_applicable_for=is_applicable_for,
        )

    def sample(self, trial: optuna.Trial) -> bool:
        """
        Sample a boolean value for the parameter during optimization.

        Args:
            trial: An Optuna trial object used to suggest parameter values.

        Returns:
            A boolean value sampled for the parameter.
        """
        return trial.suggest_categorical(self.name, [True, False])


class CategoryParameter(CpSatParameter):
    """
    A CP-SAT parameter that can take one value from a predefined list of categorical values.
    The order of these values does not have semantic significance.
    """

    def __init__(
        self,
        name: str,
        default_value,
        values: list,
        description: str = "",
        subsolver: bool = True,
        is_applicable_for: Callable[[cp_model.CpModel], bool] = _for_all_models,
    ):
        """
        Initialize the categorical parameter with a name, default value, and list of possible values.

        Args:
            name: The name of the parameter.
            default_value: The default value for the parameter.
            values: A list of possible values that the parameter can take.
        """
        super().__init__(
            name,
            default_value,
            description=description,
            is_applicable_for=is_applicable_for,
            subsolver=subsolver,
        )
        if default_value not in values:
            raise ValueError("Default value must be one of the possible values")
        self.values = values

    def sample(self, trial: optuna.Trial):
        """
        Sample a value from the list of categorical values during optimization.

        Args:
            trial: An Optuna trial object used to suggest parameter values.

        Returns:
            A value from the list of possible values.
        """
        return trial.suggest_categorical(self.name, self.values)


class IntParameter(CpSatParameter):
    """
    A CP-SAT parameter representing an integer value, which may be sampled within a defined range.
    """

    def __init__(
        self,
        name: str,
        default_value: int,
        lb: int,
        ub: int,
        log: bool,
        description: str = "",
        subsolver: bool = True,
        is_applicable_for: Callable[[cp_model.CpModel], bool] = _for_all_models,
    ):
        """
        Initialize the integer parameter with a name, default value, and range bounds.

        Args:
            name: The name of the parameter.
            default_value: The default integer value for the parameter.
            lb: The lower bound of the integer range.
            ub: The upper bound of the integer range.
            log: Whether to sample the value on a logarithmic scale.
        """
        super().__init__(
            name,
            default_value,
            description=description,
            is_applicable_for=is_applicable_for,
            subsolver=subsolver,
        )
        self.lower_bound = lb
        self.upper_bound = ub
        self.log = log

    def sample(self, trial: optuna.Trial) -> int:
        """
        Sample an integer value within the specified range during optimization.

        Args:
            trial: An Optuna trial object used to suggest parameter values.

        Returns:
            An integer value sampled within the specified range.
        """
        return trial.suggest_int(
            self.name, low=self.lower_bound, high=self.upper_bound, log=self.log
        )


class ListParameter(CpSatParameter):
    """
    A CP-SAT parameter representing a list of values, where a subset of these values must be selected.
    This parameter is split into multiple binary parameters in Optuna to facilitate optimization.
    """

    def __init__(
        self,
        name: str,
        default_value: list,
        values: list,
        description: str = "",
        subsolver: bool = True,
        is_applicable_for: Callable[[cp_model.CpModel], bool] = _for_all_models,
    ):
        """
        Initialize the list parameter with a name, default subset, and list of possible values.

        Args:
            name: The name of the parameter.
            default_value: The default subset of values.
            values: The complete list of possible values from which a subset can be selected.
        """
        super().__init__(
            name,
            tuple(sorted(default_value)),
            description=description,
            is_applicable_for=is_applicable_for,
            subsolver=subsolver,
        )
        self.values = values

    def sample(self, trial: optuna.Trial) -> list:
        """
        Sample a subset of the list of values during optimization.

        Args:
            trial: An Optuna trial object used to suggest parameter values.

        Returns:
            A list of values representing a subset of the possible values.
        """
        sampled_list = []
        for value in self.values:
            select = trial.suggest_categorical(f"{self.name}:{value}", [True, False])
            if select:
                sampled_list.append(value)
        return sampled_list

    def get_optuna_default(self) -> dict:
        """
        Retrieve the default value formatted for Optuna as a dictionary of binary selections.

        Returns:
            A dictionary representing the default selection of values in Optuna's format.
        """
        return {
            f"{self.name}:{value}": value in self._default_value
            for value in self.values
        }

    def get_cpsat_params(self, optuna_params: dict) -> dict:
        """
        Convert Optuna parameter values to CP-SAT format for this list parameter.

        Args:
            optuna_params: A dictionary of parameter values suggested by Optuna.

        Returns:
            A dictionary representing the selected subset of values in CP-SAT's format.
        """
        return {
            self.name: tuple(
                sorted(
                    value
                    for value in self.values
                    if optuna_params[f"{self.name}:{value}"]
                )
            )
        }

    def get_optuna_params(self, cpsat_params: dict) -> dict:
        """
        Convert CP-SAT parameter values to Optuna format for this list parameter.

        Args:
            cpsat_params: A dictionary of parameter values from CP-SAT.

        Returns:
            A dictionary representing the selected subset of values in Optuna's format.
        """
        return {
            f"{self.name}:{value}": value in cpsat_params[self.name]
            for value in self.values
        }


class IntFromOrderedListParameter(CpSatParameter):
    """
    A CP-SAT parameter representing an integer value selected from an ordered list of values.
    The order of the values has semantic meaning, allowing Optuna to make assumptions about this ordering during optimization.

    Note:
        The default value should be the index of the value in the list, not the value itself.
    """

    def __init__(
        self,
        name: str,
        default_index: int,
        values: list,
        description: str = "",
        subsolver: bool = True,
        is_applicable_for: Callable[[cp_model.CpModel], bool] = _for_all_models,
    ):
        """
        Initialize the parameter with a name, default index, and ordered list of possible values.

        Args:
            name: The name of the parameter.
            default_index: The index of the default value in the list of possible values.
            values: The ordered list of possible values.
        """
        super().__init__(
            name,
            default_index,
            description=description,
            is_applicable_for=is_applicable_for,
            subsolver=subsolver,
        )
        self.values = values

    def sample(self, trial: optuna.Trial):
        """
        Sample an index from the ordered list of values during optimization.

        Args:
            trial: An Optuna trial object used to suggest parameter values.

        Returns:
            The value selected from the ordered list based on the sampled index.
        """
        return self.values[
            trial.suggest_int(self.name, low=0, high=len(self.values) - 1)
        ]

    def get_optuna_default(self) -> dict:
        """
        Retrieve the default index formatted for Optuna.

        Returns:
            A dictionary representing the default index in Optuna's format.
        """
        return {self.name: self._default_value}

    def get_cpsat_default(self):
        """
        Retrieve the default value from the list formatted for CP-SAT.

        Returns:
            The value corresponding to the default index in the ordered list.
        """
        return self.values[self._default_value]

    def get_cpsat_params(self, optuna_params: dict) -> dict:
        """
        Convert Optuna parameter values to CP-SAT format for this ordered list parameter.

        Args:
            optuna_params: A dictionary of parameter values suggested by Optuna.

        Returns:
            A dictionary representing the selected value from the ordered list in CP-SAT's format.
        """
        return {self.name: self.values[optuna_params[self.name]]}

    def get_optuna_params(self, cpsat_params: dict) -> dict:
        """
        Convert CP-SAT parameter values to Optuna format for this ordered list parameter.

        Args:
            cpsat_params: A dictionary of parameter values from CP-SAT.

        Returns:
            A dictionary representing the index of the selected value in Optuna's format.
        """
        value = cpsat_params[self.name]
        return {self.name: self.values.index(value)}
