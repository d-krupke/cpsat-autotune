import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Union
from .caching_solver import CachingScorer, MultiResult
from .metrics import Comparison, Metric

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO (can be adjusted to DEBUG for more verbosity)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler()  # StreamHandler logs to console, add more handlers as needed
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Data class that stores the results of the parameter evaluation.
    """

    optimized_params: Dict[str, Union[int, bool, float, list, tuple]]
    contribution: Dict[str, float]
    optimized_score: MultiResult


class ParameterEvaluator:
    """
    Evaluates the impact of parameter changes on the model's performance.
    Identifies essential parameters and determines whether suggested parameters
    offer significant improvements over the defaults.
    """

    def __init__(
        self,
        params: Dict[str, Union[int, bool, float, list, tuple]],
        scorer: CachingScorer,
        metric: Metric,
        n_samples_for_verification: int,
        n_samples_for_trial: int,
    ) -> None:
        self.params = params
        self.scorer = scorer
        self.metric = metric
        self.results = defaultdict(list)
        self.n_samples_for_verification = n_samples_for_verification
        self.n_samples_for_trial = n_samples_for_trial
        logger.info("ParameterEvaluator initialized with params: %s", params)

    def _generate_variants(
        self, params: Dict[str, Union[int, bool, float, list, tuple]]
    ) -> Iterable[tuple[str, dict[str, Union[int, bool, float, list, tuple]]]]:
        """
        Generates solver variants by excluding one parameter at a time.
        """
        for key in params:
            reduced_params = {k: v for k, v in params.items() if k != key}
            yield key, reduced_params

    def _evaluate_single_parameter(self, key: str, params: dict) -> float:
        """
        Evaluates the impact of excluding a single parameter on the model's performance.
        """
        logger.info("Evaluating resetting parameter '%s' to default...", key)
        score = self.scorer.evaluate(params, num_runs=self.n_samples_for_trial)
        logger.debug("Score for parameter '%s': %s", key, score.mean())
        return score.mean()

    def evaluate(self) -> EvaluationResult:
        """
        Evaluates the impact of excluding each parameter individually, identifies
        the optimized set of parameters, and returns an EvaluationResult object
        containing the results.
        """
        logger.info("Starting evaluation of parameter importance...")
        default_baseline = self.scorer.evaluate(
            {}, num_runs=self.n_samples_for_verification
        )
        optuna_baseline = self.scorer.evaluate(
            self.params, num_runs=self.n_samples_for_verification
        )
        if self.metric.comp(default_baseline.mean(), optuna_baseline.mean()) in (
            Comparison.BETTER,
            Comparison.EQUAL,
        ):
            logger.warning(
                "After increasing the number of samples, no advantage was found in the optimized parameters. Discarding the results."
            )
            return EvaluationResult(
                optimized_params={}, contribution={}, optimized_score=default_baseline
            )
        accept_as_equal = (
            self.metric.worst(optuna_baseline) + optuna_baseline.mean()
        ) / 2
        optimized_params = {}
        diffs = {}

        for key, params in self._generate_variants(self.params):
            score_wo_key = self._evaluate_single_parameter(key, params)
            if self.metric.comp(score_wo_key, accept_as_equal) in (
                Comparison.EQUAL,
                Comparison.BETTER,
            ):
                logger.info(
                    "Parameter '%s' can be dropped from the optimized parameters.", key
                )
                continue
            else:
                logger.info("Parameter '%s' is essential for performance.", key)
                optimized_params[key] = self.params[key]
                diffs[key] = abs(optuna_baseline.mean() - score_wo_key)

        # Calculate parameter significance
        total_diff = sum(diffs.values())
        significance = {key: diff / total_diff for key, diff in diffs.items()}

        # Final evaluation with optimized parameters
        optimized_score = self.scorer.evaluate(
            optimized_params, num_runs=self.n_samples_for_verification
        )
        logger.debug("Optimized score: %s", optimized_score)

        if (
            self.metric.comp(
                self.metric.worst(optuna_baseline), self.metric.best(optimized_score)
            )
            == Comparison.BETTER
        ):
            # Revert to initial parameters as there seems to be some difficult correlations
            logger.warning(
                "Final evaluation shows worsened performance. Reverting to initial parameters."
            )
            optimized_params = self.params
            optimized_score = optuna_baseline
            significance = {}

        # Convert the metrics before storing them in the result
        logger.info("Evaluation complete. Returning results.")
        return EvaluationResult(
            optimized_params=optimized_params,
            contribution=significance,
            optimized_score=optimized_score,
        )
