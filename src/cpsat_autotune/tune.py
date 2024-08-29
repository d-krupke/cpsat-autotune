import logging
import optuna
from ortools.sat.python import cp_model
from .print_result import print_results
from .caching_solver import CachingScorer, MultiResult
from .objective import OptunaCpSatStrategy
from .metrics import (
    Metric,
    MinObjective,
    MaxObjective,
    MinTimeToOptimal,
    MinGapWithinTimelimit,
)
from .parameter_space import CpSatParameterSpace
from .parameter_evaluator import ParameterEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _tune(
    parameter_space: CpSatParameterSpace,
    model: cp_model.CpModel,
    metric: Metric,
    n_samples_for_verification: int,
    n_samples_for_trial: int,
    n_trials: int = 100,
) -> MultiResult:
    """
    Perform hyperparameter tuning using Optuna.

    Args:
        parameter_space (CpSatParameterSpace): The search space for parameters.
        model (cp_model.CpModel): The CP-SAT model to be optimized.
        metric (Metric): The performance metric used for evaluating the solver's results.
        n_samples_for_verification (int): The number of samples to use when verifying parameters.
        n_samples_for_trial (int): The number of samples to use for each trial.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.

    Returns:
        MultiResult: The best parameters found during the tuning process.
    """
    logger.info("Starting hyperparameter tuning with %s trials.", n_trials)
    scorer = CachingScorer(model, metric)

    # Evaluate baseline performance using default parameters
    default_baseline = scorer.evaluate({}, n_samples_for_verification)
    logger.info(
        "Baseline evaluation completed: min=%s, mean=%s, max=%s",
        default_baseline.min(),
        default_baseline.mean(),
        default_baseline.max(),
    )

    # Define the objective for the tuning process
    objective = OptunaCpSatStrategy(
        parameter_space,
        scorer=scorer,
        n_samples_for_trial=n_samples_for_trial,
        n_samples_for_verification=n_samples_for_verification,
    )

    # Initialize the study with the default parameters
    default_params = parameter_space.get_default_params_for_optuna()
    study = optuna.create_study(
        direction=objective.scorer.metric.direction,
        sampler=optuna.samplers.TPESampler(),
    )
    study.enqueue_trial(default_params)

    # Optimize the study with the defined objective function
    logger.info("Starting Optuna optimization.")
    study.optimize(objective, n_trials=n_trials)

    # Retrieve and log the best parameters
    best_params = objective.best_params()
    logger.info(
        "Best parameters found: %s. Score: %s", best_params.params, best_params.mean()
    )

    # Evaluate the best parameters and print results
    evaluator = ParameterEvaluator(
        params=best_params.params,
        scorer=scorer,
        metric=metric,
        n_samples_for_verification=n_samples_for_verification,
        n_samples_for_trial=n_samples_for_trial,
    )
    result = evaluator.evaluate()
    print_results(result, default_score=default_baseline, metric=metric)

    logger.info("Hyperparameter tuning completed.")
    return best_params


def tune_time_to_optimal(
    model: cp_model.CpModel,
    max_time_in_seconds: float,
    relative_gap_limit: float = 0.0,
    n_samples_for_trial: int = 10,
    n_samples_for_verification: int = 30,
    n_trials: int = 100,
) -> dict:
    """
    Tune CP-SAT hyperparameters to minimize the time required to find an optimal solution.

    Args:
        model (cp_model.CpModel): The CP-SAT model to be tuned for.
        max_time_in_seconds (float): The maximum time allowed for each solve operation. Set this argument
                                    to a value sufficient for the default parameters to find an optimal solution,
                                    but not much higher as it heavily influences the runtime of the tuning process.
        relative_gap_limit (float): The relative optimality gap for considering a solution as optimal.
                                    A value of 0.0 requires the solution to be exactly optimal. Often a value of
                                    0.01 or 0.001 is used to allow for small gaps, as closing the gap to 0 can often
                                    take much longer. Defaults to 0.0.
        n_samples_for_trial (int): The number of samples to take in each trial. Defaults to 10.
        n_samples_for_verification (int): The number of samples for verifying parameters. Defaults to 30.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.

    Returns:
        dict: The best parameters found during the tuning process.
    """
    logger.info("Starting tuning to minimize time to optimal solution.")

    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("use_lns_only")  # Not useful for this metric
    parameter_space.drop_parameter("max_time_in_seconds")
    parameter_space.filter_applicable_parameters([model])

    if relative_gap_limit > 0.0:
        parameter_space.drop_parameter("relative_gap_tolerance")

    metric = MinTimeToOptimal(
        max_time_in_seconds=max_time_in_seconds, relative_gap_limit=relative_gap_limit
    )

    result_params = _tune(
        parameter_space=parameter_space,
        model=model,
        metric=metric,
        n_samples_for_verification=n_samples_for_verification,
        n_samples_for_trial=n_samples_for_trial,
        n_trials=n_trials,
    ).params

    logger.info("Tuning for time to optimal completed.")
    return result_params


def tune_for_quality_within_timelimit(
    model: cp_model.CpModel,
    max_time_in_seconds: float,
    obj_for_timeout: int,
    direction: str,
    n_samples_for_trial: int = 10,
    n_samples_for_verification: int = 30,
    n_trials: int = 100,
) -> dict:
    """
    Tune CP-SAT hyperparameters to maximize or minimize solution quality within a given time limit.

    Args:
        model (cp_model.CpModel): The CP-SAT model to be tuned.
        max_time_in_seconds (float): The time limit for each solve operation in seconds. This is the
                                    time you give the solver to find a good solution. This function
                                    is useless if you set this value too high, as it should be less
                                    than the time the solver needs to find the optimal solution with
                                    the default parameters.
        obj_for_timeout (int): The objective value to return if the solver times out.
                               This should be worse than a trivial solution.
        direction (str): A string specifying whether to 'maximize' or 'minimize' the objective value.
        n_samples_for_trial (int): The number of samples to take in each trial. Defaults to 10.
        n_samples_for_verification (int): The number of samples for verifying parameters. Defaults to 30.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.

    Returns:
        dict: The best parameters found during the tuning process.

    Raises:
        ValueError: If the `direction` argument is not 'maximize' or 'minimize'.
    """
    logger.info(
        "Starting tuning for quality within time limit. Direction: %s", direction
    )

    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("max_time_in_seconds")
    parameter_space.filter_applicable_parameters([model])
    if direction == "maximize":
        metric = MaxObjective(
            obj_for_timeout=obj_for_timeout, max_time_in_seconds=max_time_in_seconds
        )
    elif direction == "minimize":
        metric = MinObjective(
            obj_for_timeout=obj_for_timeout, max_time_in_seconds=max_time_in_seconds
        )
    else:
        logger.error(
            "Invalid direction '%s'. Must be 'maximize' or 'minimize'.", direction
        )
        raise ValueError(
            "Invalid direction '%s'. Must be 'maximize' or 'minimize'." % direction
        )

    result_params = _tune(
        parameter_space=parameter_space,
        model=model,
        metric=metric,
        n_samples_for_verification=n_samples_for_verification,
        n_samples_for_trial=n_samples_for_trial,
        n_trials=n_trials,
    ).params

    logger.info("Tuning for quality within time limit completed.")
    return result_params


def tune_for_gap_within_timelimit(
    model: cp_model.CpModel,
    max_time_in_seconds: float,
    n_samples_for_trial: int = 10,
    n_samples_for_verification: int = 30,
    n_trials: int = 100,
    limit: float = 10,
) -> dict:
    """
    Tune CP-SAT hyperparameters to minimize the gap within a given time limit. This is a good
    option for more complex models for which you have no chance of finding the optimal solution
    within the time limit, but you still want to have some guarantee on the quality of the solution.
    This can be considered as a proxy for the time to optimal solution.

    CAVEAT: If the time limit is too small, it will probably only minimize the presolve time, which
    can have negative effects on the long-term performance of the solver.

    Args:
        model (cp_model.CpModel): The CP-SAT model to be tuned.
        max_time_in_seconds (float): The time limit for each solve operation in seconds. It should be set
        to value where the solver with default parameters is able to find a first reasonable but not optimal
        solution. You can also try to set it to lower values.
        n_samples_for_trial (int): The number of samples to take in each trial. Defaults to 10.
        n_samples_for_verification (int): The number of samples for verifying parameters. Defaults to 30.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.
        limit (float): The limit for the gap. Defaults to 10. 10 should be a reasonable value for most cases,
        but if the solver with default parameters is not able to find a solution with that gap within the
        time limit, you should increase it.
    """
    logger.info("Starting tuning for gap within time limit. Limit: %s", limit)

    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("max_time_in_seconds")
    parameter_space.filter_applicable_parameters([model])
    metric = MinGapWithinTimelimit(max_time_in_seconds=max_time_in_seconds, limit=limit)
    return _tune(
        parameter_space,
        model,
        metric,
        n_samples_for_verification,
        n_samples_for_trial,
        n_trials,
    ).params
