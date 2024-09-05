import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import random
from typing import Iterable, Callable, TypeVar, Any
from ortools.sat.python import cp_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


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
            logger.error(
                "Invalid direction '%s'. Must be 'minimize' or 'maximize'.", direction
            )
            raise ValueError("Direction must be either 'minimize' or 'maximize'.")
        self.direction = direction
        logger.info("Initialized Metric with direction: %s", direction)

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
        best_value = max(
            values, key=key if self.direction == "maximize" else (lambda x: -key(x))
        )
        logger.debug("Best value found: %s", best_value)
        return best_value

    def worst(self, values: Iterable[T], key: Callable[[T], Any] = lambda x: x) -> T:
        """
        Returns the worst value according to the metric's direction.
        """
        worst_value = min(
            values, key=key if self.direction == "maximize" else (lambda x: -key(x))
        )
        logger.debug("Worst value found: %s", worst_value)
        return worst_value

    def comp(self, a: T, b: T, key: Callable[[T], Any] = lambda x: x) -> Comparison:
        """
        Compares two values according to the metric's direction.
        """
        ka, kb = key(a), key(b)
        logger.debug("Comparing values: %s vs %s", ka, kb)
        if ka == kb:
            return Comparison.EQUAL
        if self.direction == "maximize":
            if ka > kb:
                return Comparison.BETTER
            else:
                return Comparison.WORSE
        else:
            if ka < kb:
                return Comparison.BETTER
            else:
                return Comparison.WORSE

    @abstractmethod
    def knockout_score(self) -> float:
        pass

    def unit(self) -> str | None:
        """
        Returns the unit of the metric.
        """
        return None

    @abstractmethod
    def objective_name(self) -> str:
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
        logger.info(
            "Starting solver with random_seed: %s, max_time_in_seconds: %s",
            solver.parameters.random_seed,
            self.max_time_in_seconds,
        )
        status = solver.solve(model)
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
            logger.info("Solver found a solution with objective value: %s", obj_value)
        else:
            logger.warning(
                "Solver did not find a feasible solution within the time limit."
            )
        return obj_value if obj_value is not None else self.obj_for_timeout

    def knockout_score(self) -> float:
        return self.obj_for_timeout

    def objective_name(self) -> str:
        return "Objective [MAX]"


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
        logger.info(
            "Starting solver with random_seed: %s, max_time_in_seconds: %s",
            solver.parameters.random_seed,
            self.max_time_in_seconds,
        )
        status = solver.solve(model)
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
            logger.info("Solver found a solution with objective value: %s", obj_value)
        else:
            logger.warning(
                "Solver did not find a feasible solution within the time limit."
            )
        return obj_value if obj_value is not None else self.obj_for_timeout

    def knockout_score(self) -> float:
        return self.obj_for_timeout

    def objective_name(self) -> str:
        return "Objective [MIN]"


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
        logger.info(
            "Starting solver with random_seed: %s, max_time_in_seconds: %s, relative_gap_limit: %s, absolute_gap_limit: %s",
            solver.parameters.random_seed,
            self.max_time_in_seconds,
            self.relative_gap_limit,
            self.absolute_gap_limit,
        )
        time_begin = datetime.now()
        status = solver.solve(model)
        time_end = datetime.now()
        time_in_s = (time_end - time_begin).total_seconds()
        logger.info("Solver completed in %s seconds with status: %s", time_in_s, status)
        if status == cp_model.OPTIMAL:
            return time_in_s
        else:
            logger.warning(
                "Solver did not find an optimal solution within the time limit."
            )
            return self.max_time_in_seconds * self.par_multiplier

    def knockout_score(self) -> float:
        return self.max_time_in_seconds * self.par_multiplier

    def objective_name(self) -> str:
        return "Time in seconds"

    def unit(self) -> str:
        return "s"


class MinGapWithinTimelimit(Metric):
    def __init__(self, max_time_in_seconds: float, limit: float):
        super().__init__(direction="minimize")
        self.max_time_in_seconds = max_time_in_seconds
        self.limit = limit

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        logger.info(
            "Starting solver with random_seed: %s, max_time_in_seconds: %s",
            solver.parameters.random_seed,
            self.max_time_in_seconds,
        )
        status = solver.solve(model)
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_val = solver.objective_value
            best_bound = solver.best_objective_bound
            gap = abs(obj_val - best_bound) / max(1, abs(obj_val))
            obj_value = solver.objective_value
            logger.info("Solver found a solution with objective value: %s", obj_value)
        else:
            gap = float("inf")
            logger.warning(
                "Solver did not find a feasible solution within the time limit."
            )
        return min(gap, self.limit)

    def knockout_score(self) -> float:
        return self.limit

    def objective_name(self) -> str:
        return "Relative gap"


class MinGapIntegralWithinTimelimit(Metric):
    def __init__(self, max_time_in_seconds: float, limit: float):
        super().__init__(direction="minimize")
        self.max_time_in_seconds = max_time_in_seconds
        self.limit = limit

    def __call__(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
    ) -> float:
        solver.parameters.random_seed = random.randint(0, 2**31 - 1)
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        logger.info(
            "Starting solver with random_seed: %s, max_time_in_seconds: %s",
            solver.parameters.random_seed,
            self.max_time_in_seconds,
        )
        status = solver.solve(model)
        obj_value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_value = solver.objective_value
            gap_integral = solver.response_proto.gap_integral
            logger.info("Solver found a solution with objective value: %s", obj_value)
        else:
            gap_integral = float("inf")
            logger.warning(
                "Solver did not find a feasible solution within the time limit."
            )
        return min(self.limit, gap_integral)

    def knockout_score(self) -> float:
        return self.limit

    def objective_name(self) -> str:
        return "Gap integral"
