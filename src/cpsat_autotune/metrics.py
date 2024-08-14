from abc import ABC, abstractmethod
from ortools.sat.python import cp_model
from ortools.sat import cp_model_pb2


class Metric(ABC):
    """
    A metric that describes how good a run of the solver was. Higher is better.
    """

    @abstractmethod
    def __call__(
        self,
        status: cp_model_pb2.CpSolverStatus,
        obj_value: float | None,
        time_in_s: float,
    ) -> float:
        pass


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
        status: cp_model_pb2.CpSolverStatus,
        obj_value: float | None,
        time_in_s: float,
    ) -> float:
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
        status: cp_model_pb2.CpSolverStatus,
        obj_value: float | None,
        time_in_s: float,
    ) -> float:
        if obj_value is not None:
            return -obj_value
        else:
            return -self.obj_for_timeout


class MinTimeToOptimal(Metric):
    """
    This metric minimizes the time it takes to find an optimal solution. Note that increasing the relative gap tolerance
    will actually consider all solutions with a gap of at most the given value as optimal.
    """

    def __init__(self, obj_for_timeout: int):
        self.obj_for_timeout = obj_for_timeout

    def __call__(
        self,
        status: cp_model_pb2.CpSolverStatus,
        obj_value: float | None,
        time_in_s: float,
    ) -> float:
        if status == cp_model.OPTIMAL:
            return -time_in_s
        else:
            return -self.obj_for_timeout
