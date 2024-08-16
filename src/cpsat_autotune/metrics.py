from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import random
from typing import Iterable, Callable, TypeVar, Any
from ortools.sat.python import cp_model

T = TypeVar('T')

class Comparison(Enum):
    WORSE = 0
    EQUAL = 1
    BETTER = 2

class Metric(ABC):
    """
    A metric that describes how good a run of the solver was.
    The direction indicates whether higher or lower values are better.
    """
    def __init__(self, direction: str):
        if direction not in ("minimize", "maximize"):
            raise ValueError("Direction must be either 'minimize' or 'maximize'.")
        self.direction = direction

    @abstractmethod
    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        pass
    
    def best(self, values: Iterable[T], key: Callable[[T], Any] = lambda x: x) -> T:
        """
        Returns the best value according to the metric's direction.
        """
        return max(values, key=key if self.direction == "maximize" else (lambda x: -key(x)))
        
    def worst(self, values: Iterable[T], key: Callable[[T], Any] = lambda x: x) -> T:
        """
        Returns the worst value according to the metric's direction.
        """
        return min(values, key=key if self.direction == "maximize" else (lambda x: -key(x)))
        
    def comp(self, a: T, b: T, key: Callable[[T], Any] = lambda x: x) -> Comparison:
        """
        Compares two values according to the metric's direction.
        """
        ka, kb = key(a), key(b)
        if ka == kb:
            return Comparison.EQUAL
        if self.direction == "maximize":
            if ka > kb:
                return Comparison.BETTER
            else:
                return Comparison.WORSE
        else:
            # minimize
            if ka < kb:
                return Comparison.BETTER
            else:
                return Comparison.WORSE
            
    @abstractmethod
    def knockout_score(self) -> float:
        pass


class MaxObjective(Metric):
    """
    This metric tries maximize the objective value within a time limit.
    """

    def __init__(self, max_time_in_seconds: float, obj_for_timeout: int):
        """
        Will return the objective value if a solution was found within the time limit, otherwise obj_for_timeout.
        It does not care about the status of the solver, but only if there was a feasible solution.
        :param obj_for_timeout: The value to return if the solver did not find any solution within the time limit.
        """
        super().__init__("maximize")
        self.obj_for_timeout = obj_for_timeout
        self.max_time_in_seconds = max_time_in_seconds

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        status = solver.solve(model)
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
        if obj_value is not None:
            return obj_value
        else:
            return self.obj_for_timeout
        
    def knockout_score(self) -> float:
        return self.obj_for_timeout


class MinObjective(Metric):
    """
    Like MaxObjective, but tries to minimize the objective value within a time limit.
    Because the metric is supposed to be maximized, the value is negated internally.
    """

    def __init__(self, max_time_in_seconds: float, obj_for_timeout: int):
        super().__init__("minimize")
        self.obj_for_timeout = obj_for_timeout
        self.max_time_in_seconds = max_time_in_seconds

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        status = solver.solve(model)
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
        if obj_value is not None:
            return obj_value
        else:
            return self.obj_for_timeout
        
    def knockout_score(self) -> float:
        return self.obj_for_timeout



class MinTimeToOptimal(Metric):
    """
    This metric minimizes the time it takes to find an optimal solution. Note that increasing the relative gap tolerance
    will actually consider all solutions with a gap of at most the given value as optimal.
    """

    def __init__(
        self,
        max_time_in_seconds: float,
        relative_gap_limit: float = 0.0,
        absolute_gap_limit: float = 0.0,
        par_multiplier: int = 10,
    ):
        super().__init__("minimize")
        self.relative_gap_limit = relative_gap_limit
        self.max_time_in_seconds = max_time_in_seconds
        self.absolute_gap_limit = absolute_gap_limit
        self.par_multiplier = par_multiplier

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        if self.relative_gap_limit > 0.0:
            solver.parameters.relative_gap_limit = self.relative_gap_limit
        if self.absolute_gap_limit > 0.0:
            solver.parameters.absolute_gap_limit = self.absolute_gap_limit
        time_begin = datetime.now()
        status = solver.solve(model)
        time_end = datetime.now()
        time_in_s = (time_end - time_begin).total_seconds()
        if status == cp_model.OPTIMAL:
            return time_in_s
        else:
            return self.max_time_in_seconds * self.par_multiplier
        
    def knockout_score(self) -> float:
        return self.max_time_in_seconds * self.par_multiplier