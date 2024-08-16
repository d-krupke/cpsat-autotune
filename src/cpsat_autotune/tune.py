import logging
import optuna
from ortools.sat.python import cp_model
from .print_result import print_results
from .caching_solver import CachingScorer, MultiResult
from .objective import OptunaCpSatStrategy
from .metrics import Metric, MinObjective, MaxObjective, MinTimeToOptimal
from .parameter_space import CpSatParameterSpace
from .parameter_evaluator import ParameterEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
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
        objective: An instance of OptunaCpSatStrategy defining the objective for tuning.
        parameter_space: An instance of CpSatParameterSpace defining the search space for parameters.
        n_trials: The number of trials to execute in the tuning process.

    Returns:
        The best parameters found during the tuning process.
    """
    logger.info("Starting hyperparameter tuning with %s trials.", n_trials)
    scorer = CachingScorer(model, metric)

    default_baseline = scorer.evaluate({}, n_samples_for_verification)
    logger.info(
        "Baseline evaluation completed: min=%s, mean=%s, max=%s",
        default_baseline.min(),
        default_baseline.mean(),
        default_baseline.max()
    )

    objective = OptunaCpSatStrategy(
        parameter_space,
        scorer=scorer,
        n_samples_for_trial=n_samples_for_trial,
        n_samples_for_verification=n_samples_for_verification,
    )

    # Initialize the study with the given parameter space and objective
    default_params = parameter_space.get_default_params_for_optuna()
    study = optuna.create_study(
        direction=objective.scorer.metric.direction, sampler=optuna.samplers.TPESampler()
    )
    study.enqueue_trial(default_params)

    # Optimize the study with the objective function
    logger.info("Starting Optuna optimization.")
    study.optimize(objective, n_trials=n_trials)

    # Retrieve and log the best parameters
    best_params = objective.best_params()
    logger.info(
        "Best parameters found: %s. Score: %s",
        best_params.params,
        best_params.mean()
    )

    # Evaluate the best parameters
    be = ParameterEvaluator(
        params=best_params.params,
        scorer=scorer,
        metric=metric,
        n_samples_for_verification=n_samples_for_verification,
        n_samples_for_trial=n_samples_for_trial
    )
    result = be.evaluate()
    print_results(result, default_baseline)
    
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
        model: The CP-SAT model for which the hyperparameters are tuned.
        timelimit_in_s: The time limit for each solve operation in seconds.
        opt_gap: The relative optimality gap that determines when a solution is considered optimal.
                 A value of 0.0 requires the solution to be exactly optimal.
        n_samples_per_param: The number of samples per parameter to take in each trial.
        max_samples_per_param: The maximum number of samples per parameter allowed before using the mean to improve runtime.
        n_trials: The number of trials to execute in the tuning process.
    """
    logger.info("Starting tuning for time to optimal.")
    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("use_lns_only")  # never useful for this metric
    parameter_space.drop_parameter("max_time_in_seconds")
    
    if relative_gap_limit > 0.0:
        parameter_space.drop_parameter("relative_gap_tolerance")

    metric = MinTimeToOptimal(max_time_in_seconds=max_time_in_seconds, relative_gap_limit=relative_gap_limit)

    result_params = _tune(
        parameter_space=parameter_space,
        model=model,
        metric=metric,
        n_samples_for_verification=n_samples_for_verification,
        n_samples_for_trial=n_samples_for_trial,
        n_trials=n_trials
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
        model: The CP-SAT model for which the hyperparameters are tuned.
        timelimit_in_s: The time limit for each solve operation in seconds.
        obj_for_timeout: The objective value to apply if the solver times out. Should be worse than a trivial solution.
        direction: A string specifying whether to 'maximize' or 'minimize' the objective value.
        n_samples_per_param: The number of samples per parameter to take in each trial.
        max_samples_per_param: The maximum number of samples per parameter allowed before using the mean to improve runtime.
        n_trials: The number of trials to execute in the tuning process.

    Raises:
        ValueError: If the `direction` argument is not 'maximize' or 'minimize'.
    """
    logger.info("Starting tuning for quality within time limit. Direction: %s", direction)
    parameter_space = CpSatParameterSpace()
    parameter_space.drop_parameter("max_time_in_seconds")

    if direction == "maximize":
        metric = MaxObjective(obj_for_timeout=obj_for_timeout, max_time_in_seconds=max_time_in_seconds)
    elif direction == "minimize":
        metric = MinObjective(obj_for_timeout=obj_for_timeout, max_time_in_seconds=max_time_in_seconds)
    else:
        logger.error("Invalid direction '%s'. Must be 'maximize' or 'minimize'.", direction)
        raise ValueError(
            "Invalid direction '%s'. Must be 'maximize' or 'minimize'." % direction
        )

    result_params = _tune(
        parameter_space=parameter_space,
        model=model,
        metric=metric,
        n_samples_for_verification=n_samples_for_verification,
        n_samples_for_trial=n_samples_for_trial,
        n_trials=n_trials
    ).params

    logger.info("Tuning for quality within time limit completed.")
    return result_params
