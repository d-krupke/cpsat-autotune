import logging

from .metrics import Comparison
from .caching_solver import CachingScorer, MultiResult
from .parameter_space import CpSatParameterSpace
import optuna

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO (can be adjusted to DEBUG for more verbosity)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler()  # StreamHandler logs to console, add more handlers as needed
    ],
)
logger = logging.getLogger(__name__)


class OptunaCpSatStrategy:
    """
    This class interacts with Optuna to tune the hyperparameters of a CP-SAT model.
    """

    def __init__(
        self,
        parameter_space: CpSatParameterSpace,
        scorer: CachingScorer,
        n_samples_for_trial: int = 10,
        n_samples_for_verification: int = 30,
    ):
        self.parameter_space = parameter_space
        self.n_samples_for_trial = n_samples_for_trial
        self.n_samples_for_verification = n_samples_for_verification
        self.scorer = scorer
        self.direction = scorer.metric.direction
        self.metric = scorer.metric

    def get_baseline(self) -> MultiResult:
        """
        Compute the baseline by solving the model with the default parameters.
        """
        return self.scorer.evaluate({}, self.n_samples_for_verification)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        This function is called by Optuna to evaluate a trial.
        """
        sampled_params = self.parameter_space.sample(trial)
        baseline = self.get_baseline()
        if self.direction == "minimize":
            knockout_score = baseline.max() + 0.1 * (baseline.spread())
        else:
            assert self.direction == "maximize"
            knockout_score = baseline.min() - 0.1 * (baseline.spread())
        score = self.scorer.evaluate(
            sampled_params,
            num_runs=self.n_samples_for_trial,
            knockout_score=knockout_score,
        )
        current_best = self.metric.best(self.scorer, key=lambda x: x.mean())
        if self.metric.comp(score.mean(), current_best.mean()) in (
            Comparison.BETTER,
            Comparison.EQUAL,
        ):
            logger.info(
                "The trial yielded the best result so far, increasing the number of samples to be sure."
            )
            score = self.scorer.evaluate(
                sampled_params,
                num_runs=self.n_samples_for_verification,
                knockout_score=knockout_score,
            )
        return score.mean()

    def best_params(self) -> MultiResult:
        """
        Returns the best parameters found so far. Use this function instead of the Optuna study's best_params
        function as it not only converts the parameters to the actual CP-SAT parameters, but can also give
        more information about the significance of the results.
        """
        return self.metric.best(self.scorer, key=lambda x: x.mean())
