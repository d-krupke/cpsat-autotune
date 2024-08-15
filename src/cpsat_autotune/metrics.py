from abc import ABC, abstractmethod
from datetime import datetime
import random
from ortools.sat.python import cp_model


class Metric(ABC):
    """
    A metric that describes how good a run of the solver was. Higher is better.
    """

    @abstractmethod
    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        pass

    def convert(self, value: float) -> float:
        return value


class MaxObjective(Metric):
    """
    This metric tries maximize the objective value within a time limit.
    """

    def __init__(self, obj_for_timeout: int):
        """
        Will return the objective value if a solution was found within the time limit, otherwise obj_for_timeout.
        It does not care about the status of the solver, but only if there was a feasible solution.
        :param obj_for_timeout: The value to return if the solver did not find any solution within the time limit.
        """
        self.obj_for_timeout = obj_for_timeout

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        time_begin = datetime.now()
        status = solver.solve(model)
        time_end = datetime.now()
        time_in_s = (time_end - time_begin).total_seconds()
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
        if obj_value is not None:
            return obj_value
        else:
            return self.obj_for_timeout


class MinObjective(Metric):
    """
    Like MaxObjective, but tries to minimize the objective value within a time limit.
    Because the metric is supposed to be maximized, the value is negated internally.
    """

    def __init__(self, obj_for_timeout: int):
        self.obj_for_timeout = obj_for_timeout

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        time_begin = datetime.now()
        status = solver.solve(model)
        time_end = datetime.now()
        time_in_s = (time_end - time_begin).total_seconds()
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
        if obj_value is not None:
            return -obj_value
        else:
            return -self.obj_for_timeout
        
    def convert(self, value: float) -> float:
        return -value


class MinTimeToOptimal(Metric):
    """
    This metric minimizes the time it takes to find an optimal solution. Note that increasing the relative gap tolerance
    will actually consider all solutions with a gap of at most the given value as optimal.
    """

    def __init__(self, obj_for_timeout: int):
        self.obj_for_timeout = obj_for_timeout

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        time_begin = datetime.now()
        status = solver.solve(model)
        time_end = datetime.now()
        time_in_s = (time_end - time_begin).total_seconds()
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
        if status == cp_model.OPTIMAL:
            return -time_in_s
        else:
            return -self.obj_for_timeout
        
    def convert(self, value: float) -> float:
        return -value
